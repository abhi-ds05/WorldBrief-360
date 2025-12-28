"""
Event bus implementation for publish-subscribe pattern with async support.
"""
import asyncio
import inspect
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Set
from contextvars import ContextVar
from concurrent.futures import ThreadPoolExecutor

from app.core.logging_config import logger
from app.events.event_types import EventType, Event
from app.events.event_handlers import EventHandlerRegistry


class EventBus:
    """
    Event bus for publish-subscribe pattern with async support.
    """

    _instance: Optional["EventBus"] = None
    _initialized = False

    _current_event: ContextVar[Optional[Event]] = ContextVar("current_event", default=None)
    _correlation_id: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)

    def __new__(cls) -> "EventBus":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self.registry = EventHandlerRegistry()
        self._event_queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=10_000)
        self._dead_letter_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=1_000)

        self._is_running = False
        self._worker_tasks: List[asyncio.Task] = []
        self._dead_letter_worker: Optional[asyncio.Task] = None
        self._pending_tasks: Set[asyncio.Task] = set()

        self._executor = ThreadPoolExecutor(
            max_workers=10,
            thread_name_prefix="event_bus",
        )

        self._metrics = {
            "events_published": 0,
            "events_processed": 0,
            "events_failed": 0,
            "dead_letter_events": 0,
        }

        self._initialized = True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self, num_workers: int = 3) -> None:
        if self._is_running:
            return

        self._is_running = True

        for i in range(num_workers):
            task = asyncio.create_task(
                self._process_events(),
                name=f"event_worker_{i}",
            )
            self._worker_tasks.append(task)

        self._dead_letter_worker = asyncio.create_task(
            self._process_dead_letter_queue(),
            name="dead_letter_worker",
        )

        logger.info(f"EventBus initialized with {num_workers} workers")

    async def shutdown(self, timeout: int = 30) -> None:
        if not self._is_running:
            return

        self._is_running = False
        logger.info("Shutting down EventBus")

        for task in self._worker_tasks:
            task.cancel()

        if self._dead_letter_worker:
            self._dead_letter_worker.cancel()

        try:
            await asyncio.wait_for(
                asyncio.gather(
                    *self._worker_tasks,
                    self._dead_letter_worker,
                    return_exceptions=True,
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("EventBus shutdown timed out")

        for task in list(self._pending_tasks):
            task.cancel()

        self._executor.shutdown(wait=True)

        logger.info(f"EventBus shutdown complete: {self._metrics}")

    # ------------------------------------------------------------------
    # Subscription
    # ------------------------------------------------------------------

    def subscribe(self, event_type: EventType, handler: Callable) -> str:
        handler_id = self.registry.register(event_type, handler)
        logger.debug(f"Subscribed handler {handler_id} to {event_type}")
        return handler_id

    def unsubscribe(self, event_type: EventType, handler_id: str) -> bool:
        return self.registry.unregister(event_type, handler_id)

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    async def publish(
        self,
        event_type: EventType,
        data: Any,
        correlation_id: Optional[str] = None,
    ) -> str:
        event_id = str(uuid.uuid4())
        correlation_id = correlation_id or self._correlation_id.get() or event_id

        event = Event(
            id=event_id,
            type=event_type,
            data=data,
            timestamp=datetime.utcnow(),
            correlation_id=correlation_id,
        )

        if self._event_queue.full():
            raise ValueError("Event queue is full")

        await self._event_queue.put(event)
        self._metrics["events_published"] += 1
        return event_id

    # ------------------------------------------------------------------
    # Workers
    # ------------------------------------------------------------------

    async def _process_events(self) -> None:
        worker_name = asyncio.current_task().get_name()
        logger.info(f"{worker_name} started")

        while self._is_running:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            token = self._current_event.set(event)
            corr_token = self._correlation_id.set(event.correlation_id)

            try:
                await self._process_event(event)
                self._metrics["events_processed"] += 1
            except Exception as exc:
                self._metrics["events_failed"] += 1
                await self._add_to_dead_letter(event, str(exc))
            finally:
                self._current_event.reset(token)
                self._correlation_id.reset(corr_token)
                self._event_queue.task_done()

        logger.info(f"{worker_name} stopped")

    async def _process_event(self, event: Event) -> None:
        handlers = self.registry.get_handlers(event.type)
        if not handlers:
            return

        tasks: List[asyncio.Task] = []

        for handler_id, handler in handlers:
            task = asyncio.create_task(
                self._execute_handler(handler_id, handler, event),
                name=f"handler_{handler_id}_{event.id[:8]}",
            )
            self._pending_tasks.add(task)
            tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute_handler(
        self,
        handler_id: str,
        handler: Callable,
        event: Event,
    ) -> None:
        try:
            if inspect.iscoroutinefunction(handler):
                await handler(event.data, event)
            else:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    self._executor,
                    handler,
                    event.data,
                    event,
                )
        finally:
            task = asyncio.current_task()
            if task in self._pending_tasks:
                self._pending_tasks.remove(task)

    # ------------------------------------------------------------------
    # Dead Letter Queue
    # ------------------------------------------------------------------

    async def _add_to_dead_letter(self, event: Event, error: str) -> None:
        dead_event = {
            "event": event.dict(),
            "error": error,
            "failed_at": datetime.utcnow().isoformat(),
        }

        try:
            await self._dead_letter_queue.put(dead_event)
            self._metrics["dead_letter_events"] += 1
            logger.error(f"Event {event.id} sent to DLQ: {error}")
        except Exception as exc:
            logger.critical(f"Failed to enqueue DLQ event: {exc}")

    async def _process_dead_letter_queue(self) -> None:
        logger.info("Dead-letter worker started")

        while self._is_running:
            try:
                dead_event = await asyncio.wait_for(
                    self._dead_letter_queue.get(),
                    timeout=5.0,
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            logger.error(
                f"Dead-letter event | id={dead_event['event']['id']} "
                f"type={dead_event['event']['type']} "
                f"error={dead_event['error']}"
            )

            self._dead_letter_queue.task_done()

        logger.info("Dead-letter worker stopped")

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, int]:
        return dict(self._metrics)

    def get_current_event(self) -> Optional[Event]:
        return self._current_event.get()

    def get_correlation_id(self) -> Optional[str]:
        return self._correlation_id.get()


# ----------------------------------------------------------------------
# Singleton access
# ----------------------------------------------------------------------

_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


async def publish_event(
    event_type: EventType,
    data: Any,
    correlation_id: Optional[str] = None,
) -> str:
    return await get_event_bus().publish(event_type, data, correlation_id)


def subscribe_to_event(event_type: EventType, handler: Callable) -> str:
    return get_event_bus().subscribe(event_type, handler)
