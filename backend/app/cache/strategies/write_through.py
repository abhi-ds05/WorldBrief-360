"""
Write-through cache strategy.
Writes to cache and data source simultaneously for consistency.
"""

import asyncio
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Callable
from contextlib import asynccontextmanager

from app.cache.backends import get_cache_backend

logger = logging.getLogger(__name__)


class WriteThroughStrategy:
    """
    Write-through cache strategy for data consistency.
    
    Features:
    - Simultaneous write to cache and data source
    - Read-after-write consistency
    - Batch write operations
    - Write coalescing
    - Retry and fallback mechanisms
    """
    
    def __init__(self, cache_backend=None, write_timeout: int = 30):
        """
        Initialize write-through strategy.
        
        Args:
            cache_backend: Cache backend instance
            write_timeout: Timeout for write operations in seconds
        """
        self.cache = cache_backend or get_cache_backend()
        self.write_timeout = write_timeout
        self._write_queue: Dict[str, Any] = {}
        self._write_locks: Dict[str, asyncio.Lock] = {}
        self._batch_size = 100
        self._coalesce_window = 0.1  # 100ms
        self._pending_batches: Dict[str, List[Tuple[float, Dict]]] = {}
        
        logger.info(f"Write-through strategy initialized with timeout: {write_timeout}s")
    
    async def _get_write_lock(self, key: str) -> asyncio.Lock:
        """
        Get or create write lock for a key.
        
        Args:
            key: Cache key
            
        Returns:
            Async lock for the key
        """
        if key not in self._write_locks:
            self._write_locks[key] = asyncio.Lock()
        return self._write_locks[key]
    
    async def write(
        self,
        key: str,
        value: Any,
        write_func: Callable[[str, Any], Any],
        read_func: Optional[Callable[[str], Any]] = None,
        ttl: Optional[int] = None,
        fallback_to_cache: bool = True,
    ) -> Any:
        """
        Write value using write-through strategy.
        
        Args:
            key: Cache key
            value: Value to write
            write_func: Function to write to data source
            read_func: Function to read from data source (for verification)
            ttl: Cache TTL in seconds
            fallback_to_cache: Use cache if write fails
            
        Returns:
            Written value
        """
        lock = await self._get_write_lock(key)
        
        async with lock:
            try:
                # Write to cache and data source concurrently
                write_task = asyncio.create_task(
                    self._execute_write(key, value, write_func)
                )
                cache_task = asyncio.create_task(
                    self.cache.set(key, value, ttl=ttl)
                )
                
                # Wait for both with timeout
                done, pending = await asyncio.wait(
                    [write_task, cache_task],
                    timeout=self.write_timeout,
                    return_when=asyncio.ALL_COMPLETED
                )
                
                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                
                # Check results
                write_success = False
                cache_success = False
                write_result = None
                cache_result = None
                
                for task in done:
                    if task == write_task:
                        try:
                            write_result = task.result()
                            write_success = True
                        except Exception as e:
                            logger.error(f"Write function failed for {key}: {e}")
                    elif task == cache_task:
                        try:
                            cache_result = task.result()
                            cache_success = True
                        except Exception as e:
                            logger.error(f"Cache set failed for {key}: {e}")
                
                if write_success and cache_success:
                    logger.debug(f"Write-through successful for {key}")
                    
                    # Verify with read if function provided
                    if read_func:
                        try:
                            if asyncio.iscoroutinefunction(read_func):
                                read_value = await read_func(key)
                            else:
                                read_value = read_func(key)
                            
                            if read_value != value:
                                logger.warning(
                                    f"Write verification failed for {key}. "
                                    f"Expected: {value}, Got: {read_value}"
                                )
                                # Update cache with actual value
                                await self.cache.set(key, read_value, ttl=ttl)
                                return read_value
                            
                        except Exception as e:
                            logger.error(f"Read verification failed for {key}: {e}")
                    
                    return write_result
                
                elif write_success and not cache_success:
                    logger.warning(f"Cache write failed but data source write succeeded for {key}")
                    # Try to update cache from data source
                    if read_func:
                        try:
                            if asyncio.iscoroutinefunction(read_func):
                                actual_value = await read_func(key)
                            else:
                                actual_value = read_func(key)
                            
                            await self.cache.set(key, actual_value, ttl=ttl)
                            return actual_value
                        except Exception as e:
                            logger.error(f"Failed to sync cache after write for {key}: {e}")
                    
                    return write_result
                
                elif not write_success and cache_success and fallback_to_cache:
                    logger.warning(f"Data source write failed, using cache for {key}")
                    # Rollback cache
                    await self.cache.delete(key)
                    raise Exception(f"Write failed for {key}")
                
                else:
                    logger.error(f"Both cache and data source write failed for {key}")
                    raise Exception(f"Write failed for {key}")
                
            except asyncio.TimeoutError:
                logger.error(f"Write timeout for {key}")
                raise TimeoutError(f"Write operation timeout for {key}")
            except Exception as e:
                logger.error(f"Write error for {key}: {e}")
                raise
    
    async def _execute_write(self, key: str, value: Any, write_func: Callable) -> Any:
        """
        Execute write function with retry logic.
        
        Args:
            key: Cache key
            value: Value to write
            write_func: Write function
            
        Returns:
            Write result
        """
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                if asyncio.iscoroutinefunction(write_func):
                    return await write_func(key, value)
                else:
                    return write_func(key, value)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                
                logger.warning(
                    f"Write attempt {attempt + 1} failed for {key}: {e}. "
                    f"Retrying in {retry_delay}s..."
                )
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
    
    async def batch_write(
        self,
        items: Dict[str, Any],
        write_func: Callable[[Dict[str, Any]], Dict[str, Any]],
        read_func: Optional[Callable[[List[str]], Dict[str, Any]]] = None,
        ttl: Optional[int] = None,
        coalesce: bool = True,
    ) -> Dict[str, Any]:
        """
        Batch write using write-through strategy.
        
        Args:
            items: Dictionary of key-value pairs
            write_func: Function to write batch to data source
            read_func: Function to read batch from data source
            ttl: Cache TTL in seconds
            coalesce: Coalesce writes within time window
            
        Returns:
            Dictionary of write results
        """
        if not items:
            return {}
        
        if coalesce:
            return await self._coalesced_batch_write(items, write_func, read_func, ttl)
        
        # Get locks for all keys
        locks = [await self._get_write_lock(key) for key in items.keys()]
        
        # Acquire all locks
        for lock in locks:
            await lock.acquire()
        
        try:
            # Write to cache
            cache_task = asyncio.create_task(
                self.cache.set_many(items, ttl=ttl)
            )
            
            # Write to data source
            write_task = asyncio.create_task(
                self._execute_batch_write(items, write_func)
            )
            
            # Wait for both
            done, pending = await asyncio.wait(
                [cache_task, write_task],
                timeout=self.write_timeout,
                return_when=asyncio.ALL_COMPLETED
            )
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
            
            # Check results
            write_results = {}
            cache_success = False
            write_success = False
            
            for task in done:
                if task == write_task:
                    try:
                        write_results = task.result()
                        write_success = True
                    except Exception as e:
                        logger.error(f"Batch write function failed: {e}")
                elif task == cache_task:
                    try:
                        cache_success = task.result()
                    except Exception as e:
                        logger.error(f"Batch cache set failed: {e}")
            
            if write_success and cache_success:
                logger.debug(f"Batch write-through successful for {len(items)} items")
                
                # Verify with read if function provided
                if read_func:
                    try:
                        if asyncio.iscoroutinefunction(read_func):
                            read_values = await read_func(list(items.keys()))
                        else:
                            read_values = read_func(list(items.keys()))
                        
                        # Check for discrepancies
                        discrepancies = []
                        for key, expected in items.items():
                            actual = read_values.get(key)
                            if actual != expected:
                                discrepancies.append((key, expected, actual))
                        
                        if discrepancies:
                            logger.warning(
                                f"Batch write verification found {len(discrepancies)} discrepancies"
                            )
                            # Update cache with actual values
                            await self.cache.set_many(read_values, ttl=ttl)
                            return read_values
                        
                    except Exception as e:
                        logger.error(f"Batch read verification failed: {e}")
                
                return write_results
            
            elif write_success and not cache_success:
                logger.warning("Batch cache write failed but data source write succeeded")
                # Try to sync cache from data source
                if read_func:
                    try:
                        if asyncio.iscoroutinefunction(read_func):
                            actual_values = await read_func(list(items.keys()))
                        else:
                            actual_values = read_func(list(items.keys()))
                        
                        await self.cache.set_many(actual_values, ttl=ttl)
                        return actual_values
                    except Exception as e:
                        logger.error(f"Failed to sync cache after batch write: {e}")
                
                return write_results
            
            else:
                logger.error("Both cache and data source batch write failed")
                raise Exception("Batch write failed")
                
        finally:
            # Release all locks
            for lock in locks:
                lock.release()
    
    async def _coalesced_batch_write(
        self,
        items: Dict[str, Any],
        write_func: Callable,
        read_func: Optional[Callable],
        ttl: Optional[int],
    ) -> Dict[str, Any]:
        """
        Coalesce writes within time window to reduce load.
        
        Args:
            items: Items to write
            write_func: Write function
            read_func: Read function
            ttl: Cache TTL
            
        Returns:
            Write results
        """
        batch_id = str(time.time())
        window_start = time.time()
        
        # Add to pending batch
        if batch_id not in self._pending_batches:
            self._pending_batches[batch_id] = []
        
        self._pending_batches[batch_id].append((window_start, items))
        
        # Wait for coalescing window
        await asyncio.sleep(self._coalesce_window)
        
        # Get all items in this batch
        all_items = {}
        for _, batch_items in self._pending_batches[batch_id]:
            all_items.update(batch_items)
        
        # Remove from pending
        del self._pending_batches[batch_id]
        
        # Execute batch write
        return await self.batch_write(
            items=all_items,
            write_func=write_func,
            read_func=read_func,
            ttl=ttl,
            coalesce=False,  # Don't coalesce recursively
        )
    
    async def _execute_batch_write(
        self,
        items: Dict[str, Any],
        write_func: Callable,
    ) -> Dict[str, Any]:
        """
        Execute batch write function.
        
        Args:
            items: Items to write
            write_func: Write function
            
        Returns:
            Write results
        """
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                if asyncio.iscoroutinefunction(write_func):
                    return await write_func(items)
                else:
                    return write_func(items)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                
                logger.warning(
                    f"Batch write attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {retry_delay}s..."
                )
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
    
    async def read(
        self,
        key: str,
        read_func: Callable[[str], Any],
        write_on_miss: bool = True,
        ttl: Optional[int] = None,
    ) -> Any:
        """
        Read value with write-through consistency.
        
        Args:
            key: Cache key
            read_func: Function to read from data source
            write_on_miss: Write to cache on miss
            ttl: Cache TTL
            
        Returns:
            Value from cache or data source
        """
        # Try cache first
        cached = await self.cache.get(key)
        if cached is not None:
            logger.debug(f"Write-through cache hit: {key}")
            return cached
        
        # Read from data source
        logger.debug(f"Write-through cache miss: {key}")
        try:
            if asyncio.iscoroutinefunction(read_func):
                value = await read_func(key)
            else:
                value = read_func(key)
        except Exception as e:
            logger.error(f"Read function failed for {key}: {e}")
            raise
        
        # Write to cache if enabled
        if write_on_miss and value is not None:
            await self.cache.set(key, value, ttl=ttl)
            logger.debug(f"Written to cache on read miss: {key}")
        
        return value
    
    async def read_many(
        self,
        keys: List[str],
        read_func: Callable[[List[str]], Dict[str, Any]],
        write_on_miss: bool = True,
        ttl: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Read multiple values with write-through consistency.
        
        Args:
            keys: List of cache keys
            read_func: Function to read batch from data source
            write_on_miss: Write to cache on miss
            ttl: Cache TTL
            
        Returns:
            Dictionary of values
        """
        if not keys:
            return {}
        
        # Try cache first
        cached = await self.cache.get_many(keys)
        result = cached.copy()
        
        # Find missing keys
        missing_keys = [k for k in keys if k not in cached]
        
        if missing_keys:
            logger.debug(f"Write-through cache misses: {len(missing_keys)} keys")
            
            # Read missing from data source
            try:
                if asyncio.iscoroutinefunction(read_func):
                    missing_values = await read_func(missing_keys)
                else:
                    missing_values = read_func(missing_keys)
            except Exception as e:
                logger.error(f"Batch read function failed: {e}")
                raise
            
            # Update result
            result.update(missing_values)
            
            # Write to cache if enabled
            if write_on_miss and missing_values:
                await self.cache.set_many(missing_values, ttl=ttl)
                logger.debug(f"Written to cache on read miss: {len(missing_values)} items")
        
        return result
    
    @asynccontextmanager
    async def transaction(self, keys: List[str]):
        """
        Context manager for write-through transaction.
        
        Args:
            keys: List of keys involved in transaction
            
        Yields:
            Transaction context
        """
        # Get locks for all keys
        locks = [await self._get_write_lock(key) for key in keys]
        
        # Acquire all locks
        for lock in locks:
            await lock.acquire()
        
        try:
            yield
        finally:
            # Release all locks
            for lock in locks:
                lock.release()
    
    async def invalidate(self, key: str, write_func: Optional[Callable] = None) -> bool:
        """
        Invalidate cache entry with optional data source update.
        
        Args:
            key: Cache key
            write_func: Optional function to update data source
            
        Returns:
            True if successful
        """
        lock = await self._get_write_lock(key)
        
        async with lock:
            if write_func:
                # Read current value
                current = await self.cache.get(key)
                if current is not None:
                    try:
                        # Update data source
                        if asyncio.iscoroutinefunction(write_func):
                            await write_func(key, None)  # Signal deletion
                        else:
                            write_func(key, None)
                    except Exception as e:
                        logger.error(f"Data source update failed during invalidation: {e}")
            
            # Delete from cache
            success = await self.cache.delete(key)
            
            if success:
                logger.debug(f"Invalidated cache entry: {key}")
            
            return success
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get write-through strategy statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "write_locks": len(self._write_locks),
            "pending_batches": len(self._pending_batches),
            "config": {
                "write_timeout": self.write_timeout,
                "batch_size": self._batch_size,
                "coalesce_window": self._coalesce_window,
            }
        }