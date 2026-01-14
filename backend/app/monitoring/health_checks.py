"""
Health Check System

This module provides a comprehensive health checking system for monitoring
the health and status of various components in the WorldBrief 360 application.
It supports:

- Synchronous and asynchronous health checks
- Dependency-based health aggregation
- Health status with detailed component information
- Health check registration and execution
- Health check scheduling and caching
- Integration with Kubernetes liveness/readiness probes
"""

import asyncio
import time
import inspect
import threading
from typing import Dict, List, Optional, Any, Callable, Union, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from functools import wraps
import json
import socket
import psutil
import redis
import sqlalchemy
from sqlalchemy import text
import httpx

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"
    
    @property
    def is_healthy(self) -> bool:
        """Check if status is healthy."""
        return self == HealthStatus.HEALTHY
    
    @property
    def is_unhealthy(self) -> bool:
        """Check if status is unhealthy."""
        return self == HealthStatus.UNHEALTHY
    
    @property
    def priority(self) -> int:
        """Get priority for status comparison (lower is worse)."""
        priorities = {
            HealthStatus.UNHEALTHY: 0,
            HealthStatus.DEGRADED: 1,
            HealthStatus.UNKNOWN: 2,
            HealthStatus.HEALTHY: 3,
        }
        return priorities.get(self, 2)
    
    @classmethod
    def from_bool(cls, is_healthy: bool) -> 'HealthStatus':
        """Convert boolean to HealthStatus."""
        return cls.HEALTHY if is_healthy else cls.UNHEALTHY
    
    @classmethod
    def aggregate(cls, statuses: List['HealthStatus']) -> 'HealthStatus':
        """Aggregate multiple health statuses."""
        if not statuses:
            return cls.UNKNOWN
        
        # Get the worst status
        worst_status = min(statuses, key=lambda s: s.priority)
        return worst_status


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    status: HealthStatus
    component: str
    check_name: str
    message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_ms: float = 0.0
    error: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "status": self.status.value,
            "component": self.component,
            "check_name": self.check_name,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "tags": list(self.tags),
        }
        
        if self.error:
            result["error"] = self.error
        
        return result
    
    @property
    def is_healthy(self) -> bool:
        """Check if result indicates healthy status."""
        return self.status.is_healthy


class HealthCheckMetadata:
    """Metadata for a health check."""
    
    def __init__(
        self,
        name: str,
        component: str,
        critical: bool = True,
        timeout: float = 5.0,
        run_on_startup: bool = True,
        tags: Optional[Set[str]] = None,
        dependencies: Optional[List[str]] = None,
        cache_ttl: Optional[float] = None,
        retry_count: int = 0,
        retry_delay: float = 0.1,
    ):
        self.name = name
        self.component = component
        self.critical = critical
        self.timeout = timeout
        self.run_on_startup = run_on_startup
        self.tags = tags or set()
        self.dependencies = dependencies or []
        self.cache_ttl = cache_ttl
        self.retry_count = retry_count
        self.retry_delay = retry_delay
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "component": self.component,
            "critical": self.critical,
            "timeout": self.timeout,
            "run_on_startup": self.run_on_startup,
            "tags": list(self.tags),
            "dependencies": self.dependencies,
            "cache_ttl": self.cache_ttl,
            "retry_count": self.retry_count,
            "retry_delay": self.retry_delay,
        }


class HealthCheck:
    """Base class for health checks."""
    
    def __init__(
        self,
        name: str,
        component: str,
        critical: bool = True,
        timeout: float = 5.0,
        run_on_startup: bool = True,
        tags: Optional[Set[str]] = None,
        dependencies: Optional[List[str]] = None,
        cache_ttl: Optional[float] = None,
        retry_count: int = 0,
        retry_delay: float = 0.1,
    ):
        self.metadata = HealthCheckMetadata(
            name=name,
            component=component,
            critical=critical,
            timeout=timeout,
            run_on_startup=run_on_startup,
            tags=tags,
            dependencies=dependencies,
            cache_ttl=cache_ttl,
            retry_count=retry_count,
            retry_delay=retry_delay,
        )
    
    async def check(self) -> HealthCheckResult:
        """Execute the health check (asynchronous)."""
        raise NotImplementedError("Subclasses must implement check()")
    
    def check_sync(self) -> HealthCheckResult:
        """Execute the health check (synchronous)."""
        raise NotImplementedError("Subclasses must implement check_sync()")
    
    @property
    def name(self) -> str:
        """Get health check name."""
        return self.metadata.name
    
    @property
    def component(self) -> str:
        """Get component name."""
        return self.metadata.component
    
    @property
    def is_critical(self) -> bool:
        """Check if health check is critical."""
        return self.metadata.critical


class FunctionHealthCheck(HealthCheck):
    """Health check based on a function."""
    
    def __init__(self, func: Callable, **kwargs):
        super().__init__(**kwargs)
        self.func = func
        self.is_async = inspect.iscoroutinefunction(func)
    
    async def check(self) -> HealthCheckResult:
        """Execute the health check."""
        start_time = time.time()
        
        try:
            if self.is_async:
                result = await asyncio.wait_for(
                    self.func(),
                    timeout=self.metadata.timeout
                )
            else:
                # Run synchronous function in thread pool
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as executor:
                    result = await asyncio.wait_for(
                        loop.run_in_executor(executor, self.func),
                        timeout=self.metadata.timeout
                    )
            
            duration_ms = (time.time() - start_time) * 1000
            
            if isinstance(result, HealthCheckResult):
                result.duration_ms = duration_ms
                return result
            elif isinstance(result, bool):
                return HealthCheckResult(
                    status=HealthStatus.from_bool(result),
                    component=self.component,
                    check_name=self.name,
                    duration_ms=duration_ms,
                    tags=self.metadata.tags,
                )
            elif isinstance(result, tuple) and len(result) == 2:
                status, message = result
                return HealthCheckResult(
                    status=HealthStatus(status) if isinstance(status, str) else status,
                    component=self.component,
                    check_name=self.name,
                    message=str(message),
                    duration_ms=duration_ms,
                    tags=self.metadata.tags,
                )
            else:
                return HealthCheckResult(
                    status=HealthStatus.HEALTHY,
                    component=self.component,
                    check_name=self.name,
                    message=str(result),
                    duration_ms=duration_ms,
                    tags=self.metadata.tags,
                )
                
        except asyncio.TimeoutError:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                component=self.component,
                check_name=self.name,
                message=f"Health check timed out after {self.metadata.timeout}s",
                duration_ms=duration_ms,
                error="timeout",
                tags=self.metadata.tags,
            )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                component=self.component,
                check_name=self.name,
                message=f"Health check failed: {str(e)}",
                duration_ms=duration_ms,
                error=str(e),
                tags=self.metadata.tags,
            )
    
    def check_sync(self) -> HealthCheckResult:
        """Execute synchronously."""
        start_time = time.time()
        
        try:
            if self.is_async:
                # Run async function in event loop
                result = asyncio.run(self.check())
            else:
                result = self.func()
                duration_ms = (time.time() - start_time) * 1000
                
                if isinstance(result, HealthCheckResult):
                    result.duration_ms = duration_ms
                    return result
                elif isinstance(result, bool):
                    return HealthCheckResult(
                        status=HealthStatus.from_bool(result),
                        component=self.component,
                        check_name=self.name,
                        duration_ms=duration_ms,
                        tags=self.metadata.tags,
                    )
                elif isinstance(result, tuple) and len(result) == 2:
                    status, message = result
                    return HealthCheckResult(
                        status=HealthStatus(status) if isinstance(status, str) else status,
                        component=self.component,
                        check_name=self.name,
                        message=str(message),
                        duration_ms=duration_ms,
                        tags=self.metadata.tags,
                    )
                else:
                    return HealthCheckResult(
                        status=HealthStatus.HEALTHY,
                        component=self.component,
                        check_name=self.name,
                        message=str(result),
                        duration_ms=duration_ms,
                        tags=self.metadata.tags,
                    )
            
            return result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                component=self.component,
                check_name=self.name,
                message=f"Health check failed: {str(e)}",
                duration_ms=duration_ms,
                error=str(e),
                tags=self.metadata.tags,
            )


# Built-in health checks
class DatabaseHealthCheck(HealthCheck):
    """Health check for database connections."""
    
    def __init__(
        self,
        engine: sqlalchemy.engine.Engine,
        component: str = "database",
        **kwargs
    ):
        super().__init__(
            name="database_connection",
            component=component,
            tags={"database", "critical"},
            **kwargs
        )
        self.engine = engine
    
    async def check(self) -> HealthCheckResult:
        """Check database connection."""
        start_time = time.time()
        
        try:
            async with self.engine.connect() as conn:
                # Execute a simple query
                result = await conn.execute(text("SELECT 1"))
                row = result.fetchone()
                
                duration_ms = (time.time() - start_time) * 1000
                
                if row and row[0] == 1:
                    return HealthCheckResult(
                        status=HealthStatus.HEALTHY,
                        component=self.component,
                        check_name=self.name,
                        message="Database connection successful",
                        details={"query_time_ms": duration_ms},
                        duration_ms=duration_ms,
                        tags=self.metadata.tags,
                    )
                else:
                    return HealthCheckResult(
                        status=HealthStatus.UNHEALTHY,
                        component=self.component,
                        check_name=self.name,
                        message="Database query returned unexpected result",
                        duration_ms=duration_ms,
                        tags=self.metadata.tags,
                    )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                component=self.component,
                check_name=self.name,
                message=f"Database connection failed: {str(e)}",
                duration_ms=duration_ms,
                error=str(e),
                tags=self.metadata.tags,
            )


class RedisHealthCheck(HealthCheck):
    """Health check for Redis connections."""
    
    def __init__(
        self,
        redis_client: redis.Redis,
        component: str = "redis",
        **kwargs
    ):
        super().__init__(
            name="redis_connection",
            component=component,
            tags={"cache", "critical"},
            **kwargs
        )
        self.redis_client = redis_client
    
    async def check(self) -> HealthCheckResult:
        """Check Redis connection."""
        start_time = time.time()
        
        try:
            # Ping Redis
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                self.redis_client.ping
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            if response:
                return HealthCheckResult(
                    status=HealthStatus.HEALTHY,
                    component=self.component,
                    check_name=self.name,
                    message="Redis connection successful",
                    details={"ping_response": response},
                    duration_ms=duration_ms,
                    tags=self.metadata.tags,
                )
            else:
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    component=self.component,
                    check_name=self.name,
                    message="Redis ping failed",
                    duration_ms=duration_ms,
                    tags=self.metadata.tags,
                )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                component=self.component,
                check_name=self.name,
                message=f"Redis connection failed: {str(e)}",
                duration_ms=duration_ms,
                error=str(e),
                tags=self.metadata.tags,
            )


class HTTPHealthCheck(HealthCheck):
    """Health check for HTTP services."""
    
    def __init__(
        self,
        url: str,
        component: str = "http_service",
        expected_status: int = 200,
        method: str = "GET",
        timeout: float = 5.0,
        **kwargs
    ):
        super().__init__(
            name="http_endpoint",
            component=component,
            timeout=timeout,
            tags={"http", "external"},
            **kwargs
        )
        self.url = url
        self.expected_status = expected_status
        self.method = method
    
    async def check(self) -> HealthCheckResult:
        """Check HTTP endpoint."""
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=self.metadata.timeout) as client:
                response = await client.request(self.method, self.url)
                
                duration_ms = (time.time() - start_time) * 1000
                
                if response.status_code == self.expected_status:
                    return HealthCheckResult(
                        status=HealthStatus.HEALTHY,
                        component=self.component,
                        check_name=self.name,
                        message=f"HTTP {self.method} {self.url} successful",
                        details={
                            "status_code": response.status_code,
                            "response_time_ms": duration_ms,
                            "content_length": len(response.content),
                        },
                        duration_ms=duration_ms,
                        tags=self.metadata.tags,
                    )
                else:
                    return HealthCheckResult(
                        status=HealthStatus.UNHEALTHY,
                        component=self.component,
                        check_name=self.name,
                        message=f"HTTP {self.method} {self.url} returned {response.status_code}",
                        details={
                            "status_code": response.status_code,
                            "response_time_ms": duration_ms,
                            "response_text": response.text[:500],
                        },
                        duration_ms=duration_ms,
                        tags=self.metadata.tags,
                    )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                component=self.component,
                check_name=self.name,
                message=f"HTTP {self.method} {self.url} failed: {str(e)}",
                duration_ms=duration_ms,
                error=str(e),
                tags=self.metadata.tags,
            )


class DiskSpaceHealthCheck(HealthCheck):
    """Health check for disk space."""
    
    def __init__(
        self,
        path: str = "/",
        threshold_percent: float = 90.0,
        component: str = "disk",
        **kwargs
    ):
        super().__init__(
            name="disk_space",
            component=component,
            tags={"system", "infrastructure"},
            **kwargs
        )
        self.path = path
        self.threshold_percent = threshold_percent
    
    async def check(self) -> HealthCheckResult:
        """Check disk space."""
        start_time = time.time()
        
        try:
            usage = psutil.disk_usage(self.path)
            used_percent = usage.percent
            
            duration_ms = (time.time() - start_time) * 1000
            
            details = {
                "path": self.path,
                "total_gb": round(usage.total / (1024**3), 2),
                "used_gb": round(usage.used / (1024**3), 2),
                "free_gb": round(usage.free / (1024**3), 2),
                "used_percent": round(used_percent, 2),
                "threshold_percent": self.threshold_percent,
            }
            
            if used_percent >= self.threshold_percent:
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    component=self.component,
                    check_name=self.name,
                    message=f"Disk usage at {used_percent:.1f}% (threshold: {self.threshold_percent}%)",
                    details=details,
                    duration_ms=duration_ms,
                    tags=self.metadata.tags,
                )
            elif used_percent >= self.threshold_percent * 0.8:  # Warning at 80% of threshold
                return HealthCheckResult(
                    status=HealthStatus.DEGRADED,
                    component=self.component,
                    check_name=self.name,
                    message=f"Disk usage at {used_percent:.1f}% (approaching threshold)",
                    details=details,
                    duration_ms=duration_ms,
                    tags=self.metadata.tags,
                )
            else:
                return HealthCheckResult(
                    status=HealthStatus.HEALTHY,
                    component=self.component,
                    check_name=self.name,
                    message=f"Disk usage at {used_percent:.1f}%",
                    details=details,
                    duration_ms=duration_ms,
                    tags=self.metadata.tags,
                )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                component=self.component,
                check_name=self.name,
                message=f"Disk space check failed: {str(e)}",
                duration_ms=duration_ms,
                error=str(e),
                tags=self.metadata.tags,
            )


class MemoryHealthCheck(HealthCheck):
    """Health check for memory usage."""
    
    def __init__(
        self,
        threshold_percent: float = 90.0,
        component: str = "memory",
        **kwargs
    ):
        super().__init__(
            name="memory_usage",
            component=component,
            tags={"system", "infrastructure"},
            **kwargs
        )
        self.threshold_percent = threshold_percent
    
    async def check(self) -> HealthCheckResult:
        """Check memory usage."""
        start_time = time.time()
        
        try:
            memory = psutil.virtual_memory()
            used_percent = memory.percent
            
            duration_ms = (time.time() - start_time) * 1000
            
            details = {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "used_percent": round(used_percent, 2),
                "threshold_percent": self.threshold_percent,
            }
            
            if used_percent >= self.threshold_percent:
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    component=self.component,
                    check_name=self.name,
                    message=f"Memory usage at {used_percent:.1f}% (threshold: {self.threshold_percent}%)",
                    details=details,
                    duration_ms=duration_ms,
                    tags=self.metadata.tags,
                )
            elif used_percent >= self.threshold_percent * 0.8:  # Warning at 80% of threshold
                return HealthCheckResult(
                    status=HealthStatus.DEGRADED,
                    component=self.component,
                    check_name=self.name,
                    message=f"Memory usage at {used_percent:.1f}% (approaching threshold)",
                    details=details,
                    duration_ms=duration_ms,
                    tags=self.metadata.tags,
                )
            else:
                return HealthCheckResult(
                    status=HealthStatus.HEALTHY,
                    component=self.component,
                    check_name=self.name,
                    message=f"Memory usage at {used_percent:.1f}%",
                    details=details,
                    duration_ms=duration_ms,
                    tags=self.metadata.tags,
                )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                component=self.component,
                check_name=self.name,
                message=f"Memory check failed: {str(e)}",
                duration_ms=duration_ms,
                error=str(e),
                tags=self.metadata.tags,
            )


class CPULoadHealthCheck(HealthCheck):
    """Health check for CPU load."""
    
    def __init__(
        self,
        threshold_percent: float = 85.0,
        component: str = "cpu",
        **kwargs
    ):
        super().__init__(
            name="cpu_load",
            component=component,
            tags={"system", "infrastructure"},
            **kwargs
        )
        self.threshold_percent = threshold_percent
    
    async def check(self) -> HealthCheckResult:
        """Check CPU load."""
        start_time = time.time()
        
        try:
            # Get CPU usage over 1 second
            cpu_percent = psutil.cpu_percent(interval=1)
            
            duration_ms = (time.time() - start_time) * 1000
            
            details = {
                "cpu_count": psutil.cpu_count(),
                "cpu_percent": round(cpu_percent, 2),
                "threshold_percent": self.threshold_percent,
            }
            
            if cpu_percent >= self.threshold_percent:
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    component=self.component,
                    check_name=self.name,
                    message=f"CPU usage at {cpu_percent:.1f}% (threshold: {self.threshold_percent}%)",
                    details=details,
                    duration_ms=duration_ms,
                    tags=self.metadata.tags,
                )
            elif cpu_percent >= self.threshold_percent * 0.8:  # Warning at 80% of threshold
                return HealthCheckResult(
                    status=HealthStatus.DEGRADED,
                    component=self.component,
                    check_name=self.name,
                    message=f"CPU usage at {cpu_percent:.1f}% (approaching threshold)",
                    details=details,
                    duration_ms=duration_ms,
                    tags=self.metadata.tags,
                )
            else:
                return HealthCheckResult(
                    status=HealthStatus.HEALTHY,
                    component=self.component,
                    check_name=self.name,
                    message=f"CPU usage at {cpu_percent:.1f}%",
                    details=details,
                    duration_ms=duration_ms,
                    tags=self.metadata.tags,
                )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                component=self.component,
                check_name=self.name,
                message=f"CPU check failed: {str(e)}",
                duration_ms=duration_ms,
                error=str(e),
                tags=self.metadata.tags,
            )


class HealthCheckRegistry:
    """Registry for managing health checks."""
    
    def __init__(self, service_name: str = "worldbrief-360"):
        """
        Initialize health check registry.
        
        Args:
            service_name: Name of the service
        """
        self.service_name = service_name
        self.health_checks: Dict[str, HealthCheck] = {}
        self.health_check_results: Dict[str, HealthCheckResult] = {}
        self._lock = threading.RLock()
        self._cache_lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=10)
        
        # Statistics
        self.stats = {
            "total_checks": 0,
            "successful_checks": 0,
            "failed_checks": 0,
            "total_duration_ms": 0.0,
            "last_check_time": None,
        }
        
        logger.info(f"HealthCheckRegistry initialized for service: {service_name}")
    
    def register(self, health_check: HealthCheck) -> None:
        """
        Register a health check.
        
        Args:
            health_check: Health check to register
        """
        with self._lock:
            if health_check.name in self.health_checks:
                logger.warning(f"Health check '{health_check.name}' already registered. Overwriting.")
            
            self.health_checks[health_check.name] = health_check
            logger.info(f"Registered health check: {health_check.name} ({health_check.component})")
    
    def register_function(
        self,
        func: Callable,
        name: str,
        component: str,
        **kwargs
    ) -> None:
        """
        Register a function as a health check.
        
        Args:
            func: Function to execute as health check
            name: Name of the health check
            component: Component name
            **kwargs: Additional arguments for FunctionHealthCheck
        """
        health_check = FunctionHealthCheck(
            func=func,
            name=name,
            component=component,
            **kwargs
        )
        self.register(health_check)
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a health check.
        
        Args:
            name: Name of the health check to unregister
            
        Returns:
            bool: True if health check was unregistered, False otherwise
        """
        with self._lock:
            if name in self.health_checks:
                del self.health_checks[name]
                logger.info(f"Unregistered health check: {name}")
                return True
            return False
    
    async def run_check(self, name: str, use_cache: bool = True) -> HealthCheckResult:
        """
        Run a single health check.
        
        Args:
            name: Name of the health check to run
            use_cache: Use cached result if available
            
        Returns:
            HealthCheckResult: Result of the health check
            
        Raises:
            KeyError: If health check not found
        """
        with self._lock:
            if name not in self.health_checks:
                raise KeyError(f"Health check '{name}' not found")
            
            health_check = self.health_checks[name]
        
        # Check cache
        if use_cache and health_check.metadata.cache_ttl:
            with self._cache_lock:
                if name in self.health_check_results:
                    cached_result = self.health_check_results[name]
                    cache_age = (datetime.utcnow() - cached_result.timestamp).total_seconds()
                    
                    if cache_age < health_check.metadata.cache_ttl:
                        logger.debug(f"Using cached result for health check: {name}")
                        return cached_result
        
        # Run health check with retries
        result = None
        last_exception = None
        
        for attempt in range(health_check.metadata.retry_count + 1):
            try:
                result = await health_check.check()
                
                # Update cache
                if health_check.metadata.cache_ttl:
                    with self._cache_lock:
                        self.health_check_results[name] = result
                
                # Update statistics
                with self._lock:
                    self.stats["total_checks"] += 1
                    self.stats["total_duration_ms"] += result.duration_ms
                    self.stats["last_check_time"] = datetime.utcnow()
                    
                    if result.is_healthy:
                        self.stats["successful_checks"] += 1
                    else:
                        self.stats["failed_checks"] += 1
                
                if result.is_healthy or attempt == health_check.metadata.retry_count:
                    return result
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Health check '{name}' failed (attempt {attempt + 1}): {e}")
                
                if attempt < health_check.metadata.retry_count:
                    await asyncio.sleep(health_check.metadata.retry_delay)
        
        # All retries failed
        duration_ms = 0.0
        if result:
            duration_ms = result.duration_ms
        
        error_result = HealthCheckResult(
            status=HealthStatus.UNHEALTHY,
            component=health_check.component,
            check_name=name,
            message=f"Health check failed after {health_check.metadata.retry_count + 1} attempts",
            duration_ms=duration_ms,
            error=str(last_exception) if last_exception else "Unknown error",
            tags=health_check.metadata.tags,
        )
        
        # Update cache
        if health_check.metadata.cache_ttl:
            with self._cache_lock:
                self.health_check_results[name] = error_result
        
        # Update statistics
        with self._lock:
            self.stats["total_checks"] += 1
            self.stats["failed_checks"] += 1
            self.stats["total_duration_ms"] += duration_ms
            self.stats["last_check_time"] = datetime.utcnow()
        
        return error_result
    
    async def run_all_checks(
        self,
        include_non_critical: bool = True,
        use_cache: bool = True,
        parallel: bool = True
    ) -> Dict[str, HealthCheckResult]:
        """
        Run all registered health checks.
        
        Args:
            include_non_critical: Include non-critical health checks
            use_cache: Use cached results when available
            parallel: Run checks in parallel
            
        Returns:
            Dict[str, HealthCheckResult]: Dictionary of health check results
        """
        with self._lock:
            # Filter health checks
            checks_to_run = {}
            for name, health_check in self.health_checks.items():
                if include_non_critical or health_check.is_critical:
                    checks_to_run[name] = health_check
            
            logger.info(f"Running {len(checks_to_run)} health checks")
        
        results = {}
        
        if parallel:
            # Run checks in parallel
            tasks = []
            for name in checks_to_run:
                task = asyncio.create_task(
                    self.run_check(name, use_cache=use_cache),
                    name=f"health_check_{name}"
                )
                tasks.append((name, task))
            
            for name, task in tasks:
                try:
                    result = await task
                    results[name] = result
                except Exception as e:
                    logger.error(f"Error running health check '{name}': {e}")
                    results[name] = HealthCheckResult(
                        status=HealthStatus.UNHEALTHY,
                        component="unknown",
                        check_name=name,
                        message=f"Health check execution failed: {str(e)}",
                        error=str(e),
                    )
        else:
            # Run checks sequentially
            for name in checks_to_run:
                try:
                    result = await self.run_check(name, use_cache=use_cache)
                    results[name] = result
                except Exception as e:
                    logger.error(f"Error running health check '{name}': {e}")
                    results[name] = HealthCheckResult(
                        status=HealthStatus.UNHEALTHY,
                        component="unknown",
                        check_name=name,
                        message=f"Health check execution failed: {str(e)}",
                        error=str(e),
                    )
        
        return results
    
    def run_all_checks_sync(
        self,
        include_non_critical: bool = True,
        use_cache: bool = True
    ) -> Dict[str, HealthCheckResult]:
        """
        Run all health checks synchronously.
        
        Args:
            include_non_critical: Include non-critical health checks
            use_cache: Use cached results when available
            
        Returns:
            Dict[str, HealthCheckResult]: Dictionary of health check results
        """
        # Run in event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # Create new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.run_all_checks(
                include_non_critical=include_non_critical,
                use_cache=use_cache,
                parallel=False  # Run sequentially for sync mode
            )
        )
    
    async def get_aggregated_health(
        self,
        include_non_critical: bool = True,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Get aggregated health status.
        
        Args:
            include_non_critical: Include non-critical health checks
            use_cache: Use cached results when available
            
        Returns:
            Dict[str, Any]: Aggregated health status
        """
        results = await self.run_all_checks(
            include_non_critical=include_non_critical,
            use_cache=use_cache
        )
        
        # Calculate aggregated status
        critical_statuses = []
        non_critical_statuses = []
        
        for name, result in results.items():
            health_check = self.health_checks.get(name)
            if health_check and health_check.is_critical:
                critical_statuses.append(result.status)
            else:
                non_critical_statuses.append(result.status)
        
        # Determine overall status
        if critical_statuses:
            overall_status = HealthStatus.aggregate(critical_statuses)
        else:
            overall_status = HealthStatus.aggregate(non_critical_statuses)
        
        # Calculate component health
        component_health = {}
        for result in results.values():
            component = result.component
            if component not in component_health:
                component_health[component] = []
            component_health[component].append(result.status)
        
        for component, statuses in component_health.items():
            component_health[component] = HealthStatus.aggregate(statuses).value
        
        # Prepare response
        response = {
            "service": self.service_name,
            "status": overall_status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {
                name: result.to_dict() for name, result in results.items()
            },
            "components": component_health,
            "summary": {
                "total_checks": len(results),
                "healthy_checks": sum(1 for r in results.values() if r.is_healthy),
                "unhealthy_checks": sum(1 for r in results.values() if not r.is_healthy),
                "critical_checks": len(critical_statuses),
                "non_critical_checks": len(non_critical_statuses),
            },
            "stats": self.get_stats(),
        }
        
        return response
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics.
        
        Returns:
            Dict[str, Any]: Statistics dictionary
        """
        with self._lock:
            stats = self.stats.copy()
            stats["registered_checks"] = len(self.health_checks)
            stats["cached_results"] = len(self.health_check_results)
            
            if stats["total_checks"] > 0:
                stats["success_rate"] = (
                    stats["successful_checks"] / stats["total_checks"] * 100
                )
                stats["average_duration_ms"] = (
                    stats["total_duration_ms"] / stats["total_checks"]
                )
            else:
                stats["success_rate"] = 0.0
                stats["average_duration_ms"] = 0.0
            
            return stats
    
    def clear_cache(self) -> None:
        """Clear cached health check results."""
        with self._cache_lock:
            self.health_check_results.clear()
            logger.info("Cleared health check cache")
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self._executor.shutdown(wait=True)
        self.clear_cache()
        logger.info("HealthCheckRegistry cleaned up")


def setup_health_checks(
    service_name: str = "worldbrief-360",
    db_engine: Optional[sqlalchemy.engine.Engine] = None,
    redis_client: Optional[redis.Redis] = None,
    **kwargs
) -> HealthCheckRegistry:
    """
    Set up health checks for the service.
    
    Args:
        service_name: Name of the service
        db_engine: SQLAlchemy engine for database health check
        redis_client: Redis client for Redis health check
        **kwargs: Additional configuration
        
    Returns:
        HealthCheckRegistry: Configured health check registry
    """
    registry = HealthCheckRegistry(service_name=service_name)
    
    # Register built-in system health checks
    registry.register(DiskSpaceHealthCheck(
        path="/",
        threshold_percent=90.0,
        component="system",
        name="disk_space_root",
        critical=True,
    ))
    
    registry.register(MemoryHealthCheck(
        threshold_percent=90.0,
        component="system",
        name="memory_usage",
        critical=True,
    ))
    
    registry.register(CPULoadHealthCheck(
        threshold_percent=85.0,
        component="system",
        name="cpu_load",
        critical=True,
    ))
    
    # Register database health check if engine provided
    if db_engine:
        registry.register(DatabaseHealthCheck(
            engine=db_engine,
            component="database",
            name="database_connection",
            critical=True,
        ))
    
    # Register Redis health check if client provided
    if redis_client:
        registry.register(RedisHealthCheck(
            redis_client=redis_client,
            component="cache",
            name="redis_connection",
            critical=False,
        ))
    
    # Register custom health checks from config
    custom_checks = kwargs.get("custom_checks", [])
    for check_config in custom_checks:
        try:
            check_type = check_config.pop("type", "function")
            
            if check_type == "function":
                # Function-based health check
                func = check_config.pop("function")
                registry.register_function(func, **check_config)
            elif check_type == "http":
                # HTTP health check
                registry.register(HTTPHealthCheck(**check_config))
            elif check_type == "database":
                # Database health check
                if db_engine:
                    registry.register(DatabaseHealthCheck(engine=db_engine, **check_config))
            elif check_type == "redis":
                # Redis health check
                if redis_client:
                    registry.register(RedisHealthCheck(redis_client=redis_client, **check_config))
            else:
                logger.warning(f"Unknown health check type: {check_type}")
                
        except Exception as e:
            logger.error(f"Failed to register custom health check: {e}")
    
    logger.info(f"Health checks setup complete. Registered {len(registry.health_checks)} checks.")
    
    return registry


# FastAPI integration helper
def create_health_router(registry: HealthCheckRegistry):
    """
    Create FastAPI router for health endpoints.
    
    Args:
        registry: HealthCheckRegistry instance
        
    Returns:
        fastapi.APIRouter: FastAPI router with health endpoints
    """
    try:
        from fastapi import APIRouter, HTTPException, Depends
        from fastapi.responses import JSONResponse
        
        router = APIRouter(tags=["health"])
        
        @router.get("/health")
        async def health(
            include_non_critical: bool = True,
            use_cache: bool = True
        ) -> JSONResponse:
            """
            Get overall health status.
            """
            health_data = await registry.get_aggregated_health(
                include_non_critical=include_non_critical,
                use_cache=use_cache
            )
            
            status_code = 200 if health_data["status"] == "healthy" else 503
            return JSONResponse(content=health_data, status_code=status_code)
        
        @router.get("/health/readiness")
        async def readiness() -> JSONResponse:
            """
            Readiness probe for Kubernetes.
            """
            health_data = await registry.get_aggregated_health(
                include_non_critical=False,  # Only critical checks for readiness
                use_cache=True
            )
            
            status_code = 200 if health_data["status"] == "healthy" else 503
            return JSONResponse(content=health_data, status_code=status_code)
        
        @router.get("/health/liveness")
        async def liveness() -> JSONResponse:
            """
            Liveness probe for Kubernetes.
            """
            # Liveness check is simpler - just check if service is running
            health_data = {
                "service": registry.service_name,
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "message": "Service is alive",
            }
            
            return JSONResponse(content=health_data, status_code=200)
        
        @router.get("/health/checks")
        async def list_checks() -> JSONResponse:
            """
            List all registered health checks.
            """
            checks = []
            for name, health_check in registry.health_checks.items():
                checks.append({
                    "name": name,
                    "component": health_check.component,
                    "critical": health_check.is_critical,
                    "metadata": health_check.metadata.to_dict(),
                })
            
            return JSONResponse(content={"checks": checks})
        
        @router.get("/health/check/{check_name}")
        async def run_single_check(
            check_name: str,
            use_cache: bool = True
        ) -> JSONResponse:
            """
            Run a single health check.
            """
            try:
                result = await registry.run_check(check_name, use_cache=use_cache)
                status_code = 200 if result.is_healthy else 503
                return JSONResponse(content=result.to_dict(), status_code=status_code)
            except KeyError:
                raise HTTPException(status_code=404, detail=f"Health check '{check_name}' not found")
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @router.get("/health/stats")
        async def get_stats() -> JSONResponse:
            """
            Get health check statistics.
            """
            stats = registry.get_stats()
            return JSONResponse(content=stats)
        
        return router
        
    except ImportError:
        logger.warning("FastAPI not installed. Cannot create health router.")
        return None


# Decorator for health-checking functions
def health_check(
    name: Optional[str] = None,
    component: str = "custom",
    critical: bool = False,
    timeout: float = 5.0,
    tags: Optional[Set[str]] = None,
    cache_ttl: Optional[float] = None,
):
    """
    Decorator to register a function as a health check.
    
    Args:
        name: Name of the health check (defaults to function name)
        component: Component name
        critical: Whether the check is critical
        timeout: Timeout in seconds
        tags: Set of tags
        cache_ttl: Cache TTL in seconds
    """
    def decorator(func):
        check_name = name or func.__name__
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # This allows the function to be called normally
            return await func(*args, **kwargs)
        
        # Store metadata for registration
        wrapper._health_check_metadata = {
            "name": check_name,
            "component": component,
            "critical": critical,
            "timeout": timeout,
            "tags": tags or set(),
            "cache_ttl": cache_ttl,
        }
        wrapper._is_health_check = True
        
        return wrapper
    
    return decorator


# Example usage
if __name__ == "__main__":
    import asyncio
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    async def main():
        # Create registry
        registry = setup_health_checks(service_name="test-service")
        
        # Add a custom health check using decorator
        @health_check(name="custom_check", component="custom", critical=False)
        async def custom_health_check():
            """Custom health check example."""
            await asyncio.sleep(0.1)  # Simulate work
            return True, "Custom check passed"
        
        # Register the decorated function
        registry.register_function(
            func=custom_health_check,
            name="custom_check",
            component="custom",
            critical=False
        )
        
        # Run all checks
        print("Running all health checks...")
        results = await registry.run_all_checks()
        
        for name, result in results.items():
            print(f"{name}: {result.status.value} ({result.duration_ms:.1f}ms)")
        
        # Get aggregated health
        print("\nAggregated health:")
        health_data = await registry.get_aggregated_health()
        print(json.dumps(health_data, indent=2, default=str))
        
        # Get statistics
        print("\nStatistics:")
        stats = registry.get_stats()
        print(json.dumps(stats, indent=2, default=str))
        
        # Cleanup
        registry.cleanup()
        
        print("\nHealth check test completed")
    
    asyncio.run(main())