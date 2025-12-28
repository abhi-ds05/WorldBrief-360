"""
Background Task Subscriber for handling long-running and deferred tasks.
Listens for events that trigger background processing and delegates them to appropriate workers.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union
import json
import uuid
from enum import Enum
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from sqlalchemy import select, update, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging_config import logger
from app.db.session import AsyncSessionLocal
from app.db.models.background_task import BackgroundTask, TaskStatus, TaskPriority
from app.events.event_bus import EventBus
from app.events.event_types import EventType
from app.tasks.celery_app import celery_app
from app.services.utils.background_tasks import BackgroundTaskManager


class TaskType(str, Enum):
    """Types of background tasks."""
    DATA_INGESTION = "data_ingestion"
    DATA_PROCESSING = "data_processing"
    CONTENT_GENERATION = "content_generation"
    INCIDENT_VERIFICATION = "incident_verification"
    NOTIFICATION_SENDING = "notification_sending"
    REPORT_GENERATION = "report_generation"
    SYSTEM_CLEANUP = "system_cleanup"
    MODEL_TRAINING = "model_training"
    EMBEDDING_GENERATION = "embedding_generation"


@dataclass
class TaskConfig:
    """Configuration for a background task."""
    task_type: TaskType
    priority: TaskPriority = TaskPriority.MEDIUM
    timeout_seconds: int = 3600  # 1 hour default
    max_retries: int = 3
    retry_delay_seconds: int = 60
    dependencies: List[str] = None  # Task IDs that must complete first
    metadata: Dict[str, Any] = None


class BackgroundTaskSubscriber:
    """
    Subscriber that listens for events requiring background processing
    and manages task execution through Celery or async workers.
    """
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.task_manager = BackgroundTaskManager()
        self._subscriptions = []
        self._executor = ThreadPoolExecutor(max_workers=10)
        self._running_tasks = {}
        
    async def initialize(self):
        """Subscribe to background task events."""
        # Subscribe to data processing events
        self._subscriptions.extend([
            await self.event_bus.subscribe(EventType.DATA_INGESTION_REQUESTED, self.handle_data_ingestion),
            await self.event_bus.subscribe(EventType.DATA_PROCESSING_REQUESTED, self.handle_data_processing),
            await self.event_bus.subscribe(EventType.EMBEDDING_GENERATION_REQUESTED, self.handle_embedding_generation),
        ])
        
        # Subscribe to content generation events
        self._subscriptions.extend([
            await self.event_bus.subscribe(EventType.BRIEFING_GENERATION_REQUESTED, self.handle_briefing_generation),
            await self.event_bus.subscribe(EventType.IMAGE_GENERATION_REQUESTED, self.handle_image_generation),
            await self.event_bus.subscribe(EventType.TTS_GENERATION_REQUESTED, self.handle_tts_generation),
        ])
        
        # Subscribe to incident-related events
        self._subscriptions.extend([
            await self.event_bus.subscribe(EventType.INCIDENT_VERIFICATION_REQUESTED, self.handle_incident_verification),
            await self.event_bus.subscribe(EventType.INCIDENT_ANALYSIS_REQUESTED, self.handle_incident_analysis),
        ])
        
        # Subscribe to system events
        self._subscriptions.extend([
            await self.event_bus.subscribe(EventType.REPORT_GENERATION_REQUESTED, self.handle_report_generation),
            await self.event_bus.subscribe(EventType.SYSTEM_CLEANUP_REQUESTED, self.handle_system_cleanup),
            await self.event_bus.subscribe(EventType.MODEL_TRAINING_REQUESTED, self.handle_model_training),
        ])
        
        # Start background task monitor
        asyncio.create_task(self._monitor_tasks())
        
        logger.info("BackgroundTaskSubscriber initialized")
    
    async def cleanup(self):
        """Cleanup resources."""
        for subscription in self._subscriptions:
            await self.event_bus.unsubscribe(subscription)
        self._subscriptions.clear()
        
        # Cancel running tasks
        for task_id in list(self._running_tasks.keys()):
            await self.cancel_task(task_id)
        
        self._executor.shutdown(wait=False)
        logger.info("BackgroundTaskSubscriber cleaned up")
    
    async def _create_task_record(
        self,
        task_type: TaskType,
        user_id: Optional[int],
        input_data: Dict[str, Any],
        config: TaskConfig
    ) -> BackgroundTask:
        """
        Create a task record in the database.
        
        Args:
            task_type: Type of task
            user_id: ID of user who triggered the task
            input_data: Input data for the task
            config: Task configuration
            
        Returns:
            BackgroundTask instance
        """
        try:
            async with AsyncSessionLocal() as session:
                task_id = str(uuid.uuid4())
                
                background_task = BackgroundTask(
                    task_id=task_id,
                    task_type=task_type.value,
                    status=TaskStatus.PENDING,
                    priority=config.priority,
                    user_id=user_id,
                    input_data=json.dumps(input_data) if input_data else None,
                    output_data=None,
                    error_message=None,
                    timeout_seconds=config.timeout_seconds,
                    max_retries=config.max_retries,
                    retry_count=0,
                    dependencies=json.dumps(config.dependencies) if config.dependencies else None,
                    metadata=json.dumps(config.metadata) if config.metadata else None,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                
                session.add(background_task)
                await session.commit()
                await session.refresh(background_task)
                
                logger.info(f"Created background task {task_id} of type {task_type}")
                return background_task
                
        except Exception as e:
            logger.error(f"Failed to create task record: {e}", exc_info=True)
            raise
    
    async def _update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        output_data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        progress: Optional[float] = None
    ):
        """
        Update task status in the database.
        
        Args:
            task_id: Task identifier
            status: New status
            output_data: Output data from task execution
            error_message: Error message if failed
            progress: Progress percentage (0-100)
        """
        try:
            async with AsyncSessionLocal() as session:
                # Get the task
                result = await session.execute(
                    select(BackgroundTask).where(BackgroundTask.task_id == task_id)
                )
                task = result.scalar_one_or_none()
                
                if not task:
                    logger.warning(f"Task {task_id} not found for status update")
                    return
                
                # Update task
                task.status = status
                task.updated_at = datetime.utcnow()
                
                if output_data is not None:
                    task.output_data = json.dumps(output_data)
                
                if error_message is not None:
                    task.error_message = error_message
                    task.retry_count += 1
                
                if progress is not None:
                    task.progress = progress
                
                await session.commit()
                
                # Emit task status update event
                await self._emit_task_status_update(task)
                
                logger.debug(f"Updated task {task_id} status to {status}")
                
        except Exception as e:
            logger.error(f"Failed to update task status: {e}", exc_info=True)
    
    async def _emit_task_status_update(self, task: BackgroundTask):
        """Emit event when task status changes."""
        try:
            await self.event_bus.emit(EventType.BACKGROUND_TASK_UPDATED, {
                "task_id": task.task_id,
                "task_type": task.task_type,
                "status": task.status.value,
                "user_id": task.user_id,
                "progress": task.progress,
                "created_at": task.created_at.isoformat(),
                "updated_at": task.updated_at.isoformat(),
                "error_message": task.error_message
            })
        except Exception as e:
            logger.error(f"Failed to emit task status update: {e}")
    
    async def _execute_celery_task(
        self,
        task_name: str,
        task_id: str,
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        Execute a task through Celery.
        
        Args:
            task_name: Name of the Celery task
            task_id: Unique task identifier
            args: Positional arguments
            kwargs: Keyword arguments
        """
        try:
            # Update status to running
            await self._update_task_status(task_id, TaskStatus.RUNNING)
            
            # Execute Celery task
            celery_task = celery_app.send_task(
                task_name,
                args=args,
                kwargs=kwargs or {},
                task_id=task_id
            )
            
            # Store reference
            self._running_tasks[task_id] = celery_task
            
            logger.info(f"Started Celery task {task_name} with ID {task_id}")
            
        except Exception as e:
            logger.error(f"Failed to execute Celery task {task_name}: {e}", exc_info=True)
            await self._update_task_status(task_id, TaskStatus.FAILED, error_message=str(e))
    
    async def _execute_async_task(
        self,
        task_id: str,
        async_func: Callable,
        *args,
        **kwargs
    ):
        """
        Execute an async function as a background task.
        
        Args:
            task_id: Unique task identifier
            async_func: Async function to execute
            args: Positional arguments
            kwargs: Keyword arguments
        """
        try:
            # Update status to running
            await self._update_task_status(task_id, TaskStatus.RUNNING)
            
            # Execute async task
            task = asyncio.create_task(self._run_async_task(async_func, task_id, *args, **kwargs))
            self._running_tasks[task_id] = task
            
            logger.info(f"Started async task {async_func.__name__} with ID {task_id}")
            
        except Exception as e:
            logger.error(f"Failed to execute async task: {e}", exc_info=True)
            await self._update_task_status(task_id, TaskStatus.FAILED, error_message=str(e))
    
    async def _run_async_task(
        self,
        async_func: Callable,
        task_id: str,
        *args,
        **kwargs
    ):
        """Wrapper to run async task with proper error handling."""
        try:
            result = await async_func(*args, **kwargs)
            await self._update_task_status(task_id, TaskStatus.COMPLETED, output_data=result)
            
        except Exception as e:
            logger.error(f"Async task {task_id} failed: {e}", exc_info=True)
            await self._update_task_status(task_id, TaskStatus.FAILED, error_message=str(e))
            
        finally:
            # Remove from running tasks
            self._running_tasks.pop(task_id, None)
    
    async def _execute_sync_task(
        self,
        task_id: str,
        sync_func: Callable,
        *args,
        **kwargs
    ):
        """
        Execute a synchronous function in a thread pool.
        
        Args:
            task_id: Unique task identifier
            sync_func: Synchronous function to execute
            args: Positional arguments
            kwargs: Keyword arguments
        """
        try:
            # Update status to running
            await self._update_task_status(task_id, TaskStatus.RUNNING)
            
            # Execute in thread pool
            future = self._executor.submit(self._run_sync_task, sync_func, task_id, *args, **kwargs)
            self._running_tasks[task_id] = future
            
            logger.info(f"Started sync task {sync_func.__name__} with ID {task_id}")
            
        except Exception as e:
            logger.error(f"Failed to execute sync task: {e}", exc_info=True)
            await self._update_task_status(task_id, TaskStatus.FAILED, error_message=str(e))
    
    def _run_sync_task(
        self,
        sync_func: Callable,
        task_id: str,
        *args,
        **kwargs
    ):
        """Wrapper to run sync task with proper error handling."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = sync_func(*args, **kwargs)
            
            # Schedule status update in the main event loop
            asyncio.run_coroutine_threadsafe(
                self._update_task_status(task_id, TaskStatus.COMPLETED, output_data=result),
                asyncio.get_running_loop()
            )
            
        except Exception as e:
            logger.error(f"Sync task {task_id} failed: {e}", exc_info=True)
            
            # Schedule status update in the main event loop
            asyncio.run_coroutine_threadsafe(
                self._update_task_status(task_id, TaskStatus.FAILED, error_message=str(e)),
                asyncio.get_running_loop()
            )
            
        finally:
            loop.close()
            # Remove from running tasks (schedule in main loop)
            asyncio.run_coroutine_threadsafe(
                self._cleanup_task(task_id),
                asyncio.get_running_loop()
            )
    
    async def _cleanup_task(self, task_id: str):
        """Cleanup task from running tasks dictionary."""
        self._running_tasks.pop(task_id, None)
    
    async def _monitor_tasks(self):
        """Monitor background tasks for timeouts and retries."""
        while True:
            try:
                await self._check_timeouts()
                await self._retry_failed_tasks()
                await self._cleanup_old_tasks()
                
            except Exception as e:
                logger.error(f"Error in task monitor: {e}", exc_info=True)
            
            await asyncio.sleep(60)  # Check every minute
    
    async def _check_timeouts(self):
        """Check for tasks that have timed out."""
        try:
            async with AsyncSessionLocal() as session:
                # Find running tasks that have timed out
                timeout_threshold = datetime.utcnow() - timedelta(seconds=3600)  # Adjust as needed
                
                result = await session.execute(
                    select(BackgroundTask)
                    .where(BackgroundTask.status == TaskStatus.RUNNING)
                    .where(BackgroundTask.updated_at < timeout_threshold)
                )
                
                timed_out_tasks = result.scalars().all()
                
                for task in timed_out_tasks:
                    logger.warning(f"Task {task.task_id} timed out")
                    await self._update_task_status(
                        task.task_id,
                        TaskStatus.FAILED,
                        error_message="Task execution timed out"
                    )
                    
        except Exception as e:
            logger.error(f"Error checking timeouts: {e}")
    
    async def _retry_failed_tasks(self):
        """Retry failed tasks that haven't exceeded max retries."""
        try:
            async with AsyncSessionLocal() as session:
                # Find failed tasks eligible for retry
                result = await session.execute(
                    select(BackgroundTask)
                    .where(BackgroundTask.status == TaskStatus.FAILED)
                    .where(BackgroundTask.retry_count < BackgroundTask.max_retries)
                    .where(BackgroundTask.updated_at < datetime.utcnow() - timedelta(minutes=5))
                )
                
                retry_tasks = result.scalars().all()
                
                for task in retry_tasks:
                    # TODO: Implement retry logic based on task type
                    logger.info(f"Task {task.task_id} eligible for retry ({task.retry_count + 1}/{task.max_retries})")
                    
        except Exception as e:
            logger.error(f"Error checking retries: {e}")
    
    async def _cleanup_old_tasks(self):
        """Clean up old completed tasks."""
        try:
            async with AsyncSessionLocal() as session:
                # Find completed tasks older than 7 days
                cleanup_threshold = datetime.utcnow() - timedelta(days=7)
                
                result = await session.execute(
                    select(BackgroundTask)
                    .where(BackgroundTask.status.in_([TaskStatus.COMPLETED, TaskStatus.CANCELLED]))
                    .where(BackgroundTask.updated_at < cleanup_threshold)
                )
                
                old_tasks = result.scalars().all()
                
                # Archive or delete old tasks
                for task in old_tasks:
                    # TODO: Implement archival logic
                    logger.debug(f"Task {task.task_id} is eligible for cleanup")
                    
        except Exception as e:
            logger.error(f"Error cleaning up old tasks: {e}")
    
    # Event Handlers
    async def handle_data_ingestion(self, event):
        """Handle data ingestion requests."""
        config = TaskConfig(
            task_type=TaskType.DATA_INGESTION,
            priority=TaskPriority.HIGH if event.data.get("priority") == "high" else TaskPriority.MEDIUM,
            timeout_seconds=7200,  # 2 hours for ingestion
            metadata={
                "source": event.data.get("source"),
                "ingestion_type": event.data.get("ingestion_type")
            }
        )
        
        task = await self._create_task_record(
            TaskType.DATA_INGESTION,
            event.user_id,
            event.data,
            config
        )
        
        # Execute via Celery
        await self._execute_celery_task(
            "app.tasks.ingestion_tasks.process_data_ingestion",
            task.task_id,
            kwargs={
                "task_id": task.task_id,
                "source": event.data.get("source"),
                "ingestion_type": event.data.get("ingestion_type"),
                "parameters": event.data.get("parameters", {})
            }
        )
    
    async def handle_data_processing(self, event):
        """Handle data processing requests."""
        config = TaskConfig(
            task_type=TaskType.DATA_PROCESSING,
            metadata={
                "processing_type": event.data.get("processing_type"),
                "data_id": event.data.get("data_id")
            }
        )
        
        task = await self._create_task_record(
            TaskType.DATA_PROCESSING,
            event.user_id,
            event.data,
            config
        )
        
        # Execute via Celery
        await self._execute_celery_task(
            "app.tasks.processing_tasks.process_data",
            task.task_id,
            kwargs={
                "task_id": task.task_id,
                "data_id": event.data.get("data_id"),
                "processing_type": event.data.get("processing_type")
            }
        )
    
    async def handle_embedding_generation(self, event):
        """Handle embedding generation requests."""
        config = TaskConfig(
            task_type=TaskType.EMBEDDING_GENERATION,
            timeout_seconds=1800,  # 30 minutes
            metadata={
                "model": event.data.get("model"),
                "text_count": event.data.get("text_count", 0)
            }
        )
        
        task = await self._create_task_record(
            TaskType.EMBEDDING_GENERATION,
            event.user_id,
            event.data,
            config
        )
        
        # Execute via async task
        from app.services.processing.embedder import generate_embeddings_batch
        
        await self._execute_async_task(
            task.task_id,
            generate_embeddings_batch,
            texts=event.data.get("texts", []),
            model_name=event.data.get("model", "sentence-transformers/all-MiniLM-L6-v2"),
            task_id=task.task_id
        )
    
    async def handle_briefing_generation(self, event):
        """Handle briefing generation requests."""
        config = TaskConfig(
            task_type=TaskType.CONTENT_GENERATION,
            priority=TaskPriority.HIGH if event.data.get("urgent") else TaskPriority.MEDIUM,
            timeout_seconds=300,  # 5 minutes
            metadata={
                "topic": event.data.get("topic"),
                "level": event.data.get("level"),
                "user_preferences": event.data.get("user_preferences", {})
            }
        )
        
        task = await self._create_task_record(
            TaskType.CONTENT_GENERATION,
            event.user_id,
            event.data,
            config
        )
        
        # Execute via Celery
        await self._execute_celery_task(
            "app.tasks.generation_tasks.generate_briefing",
            task.task_id,
            kwargs={
                "task_id": task.task_id,
                "user_id": event.user_id,
                "topic": event.data.get("topic"),
                "level": event.data.get("level"),
                "options": event.data.get("options", {})
            }
        )
    
    async def handle_image_generation(self, event):
        """Handle image generation requests."""
        config = TaskConfig(
            task_type=TaskType.CONTENT_GENERATION,
            timeout_seconds=600,  # 10 minutes
            metadata={
                "model": event.data.get("model"),
                "prompt": event.data.get("prompt"),
                "style": event.data.get("style")
            }
        )
        
        task = await self._create_task_record(
            TaskType.CONTENT_GENERATION,
            event.user_id,
            event.data,
            config
        )
        
        # Execute via Celery
        await self._execute_celery_task(
            "app.tasks.generation_tasks.generate_image",
            task.task_id,
            kwargs={
                "task_id": task.task_id,
                "prompt": event.data.get("prompt"),
                "model": event.data.get("model", "stabilityai/stable-diffusion-2-1"),
                "style": event.data.get("style"),
                "user_id": event.user_id
            }
        )
    
    async def handle_tts_generation(self, event):
        """Handle text-to-speech generation requests."""
        config = TaskConfig(
            task_type=TaskType.CONTENT_GENERATION,
            timeout_seconds=300,  # 5 minutes
            metadata={
                "text_length": len(event.data.get("text", "")),
                "voice": event.data.get("voice")
            }
        )
        
        task = await self._create_task_record(
            TaskType.CONTENT_GENERATION,
            event.user_id,
            event.data,
            config
        )
        
        # Execute via async task
        from app.services.generators.tts_generator import generate_speech
        
        await self._execute_async_task(
            task.task_id,
            generate_speech,
            text=event.data.get("text"),
            voice=event.data.get("voice", "en-US-Studio-O"),
            task_id=task.task_id
        )
    
    async def handle_incident_verification(self, event):
        """Handle incident verification requests."""
        config = TaskConfig(
            task_type=TaskType.INCIDENT_VERIFICATION,
            priority=TaskPriority.HIGH,  # Usually high priority
            timeout_seconds=900,  # 15 minutes
            metadata={
                "incident_id": event.data.get("incident_id"),
                "verification_type": event.data.get("verification_type")
            }
        )
        
        task = await self._create_task_record(
            TaskType.INCIDENT_VERIFICATION,
            event.user_id,
            event.data,
            config
        )
        
        # Execute via Celery
        await self._execute_celery_task(
            "app.tasks.verification_tasks.verify_incident",
            task.task_id,
            kwargs={
                "task_id": task.task_id,
                "incident_id": event.data.get("incident_id"),
                "verification_type": event.data.get("verification_type"),
                "user_id": event.user_id
            }
        )
    
    async def handle_incident_analysis(self, event):
        """Handle incident analysis requests."""
        config = TaskConfig(
            task_type=TaskType.INCIDENT_VERIFICATION,
            timeout_seconds=1200,  # 20 minutes
            metadata={
                "incident_id": event.data.get("incident_id"),
                "analysis_type": event.data.get("analysis_type")
            }
        )
        
        task = await self._create_task_record(
            TaskType.INCIDENT_VERIFICATION,
            event.user_id,
            event.data,
            config
        )
        
        # Execute via async task
        from app.services.community.verification_engine import analyze_incident
        
        await self._execute_async_task(
            task.task_id,
            analyze_incident,
            incident_id=event.data.get("incident_id"),
            analysis_type=event.data.get("analysis_type"),
            task_id=task.task_id
        )
    
    async def handle_report_generation(self, event):
        """Handle report generation requests."""
        config = TaskConfig(
            task_type=TaskType.REPORT_GENERATION,
            timeout_seconds=1800,  # 30 minutes
            metadata={
                "report_type": event.data.get("report_type"),
                "date_range": event.data.get("date_range")
            }
        )
        
        task = await self._create_task_record(
            TaskType.REPORT_GENERATION,
            event.user_id,
            event.data,
            config
        )
        
        # Execute via sync task (CPU intensive)
        from app.services.analytics.report_generator import generate_report
        
        await self._execute_sync_task(
            task.task_id,
            generate_report,
            report_type=event.data.get("report_type"),
            start_date=event.data.get("start_date"),
            end_date=event.data.get("end_date"),
            parameters=event.data.get("parameters", {})
        )
    
    async def handle_system_cleanup(self, event):
        """Handle system cleanup requests."""
        config = TaskConfig(
            task_type=TaskType.SYSTEM_CLEANUP,
            priority=TaskPriority.LOW,
            timeout_seconds=3600,  # 1 hour
            metadata={
                "cleanup_type": event.data.get("cleanup_type"),
                "retention_days": event.data.get("retention_days", 30)
            }
        )
        
        task = await self._create_task_record(
            TaskType.SYSTEM_CLEANUP,
            None,  # System task, no user
            event.data,
            config
        )
        
        # Execute via Celery
        await self._execute_celery_task(
            "app.tasks.cleanup_tasks.cleanup_system_data",
            task.task_id,
            kwargs={
                "task_id": task.task_id,
                "cleanup_type": event.data.get("cleanup_type"),
                "retention_days": event.data.get("retention_days", 30)
            }
        )
    
    async def handle_model_training(self, event):
        """Handle model training requests."""
        config = TaskConfig(
            task_type=TaskType.MODEL_TRAINING,
            priority=TaskPriority.LOW,  # Usually low priority
            timeout_seconds=86400,  # 24 hours
            max_retries=1,  # Training usually not retried
            metadata={
                "model_type": event.data.get("model_type"),
                "training_data_size": event.data.get("training_data_size", 0)
            }
        )
        
        task = await self._create_task_record(
            TaskType.MODEL_TRAINING,
            event.user_id,
            event.data,
            config
        )
        
        # Execute via sync task in process pool (CPU intensive)
        from app.models.text_generation.factory import train_model
        
        await self._execute_sync_task(
            task.task_id,
            train_model,
            model_type=event.data.get("model_type"),
            training_data=event.data.get("training_data"),
            parameters=event.data.get("parameters", {})
        )
    
    # Task Management Methods
    async def get_task_status(self, task_id: str) -> Optional[BackgroundTask]:
        """Get status of a specific task."""
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(BackgroundTask).where(BackgroundTask.task_id == task_id)
                )
                return result.scalar_one_or_none()
                
        except Exception as e:
            logger.error(f"Error getting task status: {e}")
            return None
    
    async def get_user_tasks(
        self,
        user_id: int,
        status: Optional[TaskStatus] = None,
        task_type: Optional[TaskType] = None,
        limit: int = 50
    ) -> List[BackgroundTask]:
        """Get tasks for a specific user."""
        try:
            async with AsyncSessionLocal() as session:
                query = select(BackgroundTask).where(BackgroundTask.user_id == user_id)
                
                if status:
                    query = query.where(BackgroundTask.status == status)
                
                if task_type:
                    query = query.where(BackgroundTask.task_type == task_type.value)
                
                query = query.order_by(desc(BackgroundTask.created_at)).limit(limit)
                
                result = await session.execute(query)
                return result.scalars().all()
                
        except Exception as e:
            logger.error(f"Error getting user tasks: {e}")
            return []
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        try:
            # Get task from running tasks
            task = self._running_tasks.get(task_id)
            
            if task:
                # Cancel based on task type
                if isinstance(task, asyncio.Task):
                    task.cancel()
                elif hasattr(task, 'revoke'):  # Celery task
                    task.revoke(terminate=True)
                elif hasattr(task, 'cancel'):  # Future
                    task.cancel()
            
            # Update status in database
            await self._update_task_status(task_id, TaskStatus.CANCELLED)
            
            # Remove from running tasks
            self._running_tasks.pop(task_id, None)
            
            logger.info(f"Cancelled task {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling task {task_id}: {e}")
            return False
    
    async def retry_task(self, task_id: str) -> bool:
        """Retry a failed task."""
        try:
            # Get task from database
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(BackgroundTask).where(BackgroundTask.task_id == task_id)
                )
                task = result.scalar_one_or_none()
                
                if not task:
                    logger.warning(f"Task {task_id} not found for retry")
                    return False
                
                # Check if retry is allowed
                if task.retry_count >= task.max_retries:
                    logger.warning(f"Task {task_id} has exceeded max retries")
                    return False
                
                # Reset task for retry
                task.status = TaskStatus.PENDING
                task.updated_at = datetime.utcnow()
                await session.commit()
                
                # TODO: Implement retry logic based on task type
                logger.info(f"Task {task_id} marked for retry")
                return True
                
        except Exception as e:
            logger.error(f"Error retrying task {task_id}: {e}")
            return False


# Factory function
async def create_background_task_subscriber(event_bus: EventBus) -> BackgroundTaskSubscriber:
    """Create and initialize a background task subscriber."""
    subscriber = BackgroundTaskSubscriber(event_bus)
    await subscriber.initialize()
    return subscriber