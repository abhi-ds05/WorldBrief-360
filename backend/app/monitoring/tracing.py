"""
Distributed Tracing System

This module provides distributed tracing capabilities with:

- OpenTelemetry integration for standards-based tracing
- Automatic instrumentation for HTTP requests, database calls, etc.
- Custom span creation for business logic
- Trace context propagation across services
- Trace sampling and filtering
- Trace exporting to various backends (Jaeger, Zipkin, etc.)
- Trace analysis and visualization support
- Performance analysis through spans
- Error tracking with trace context
"""

import time
import threading
import asyncio
import uuid
import inspect
import functools
from typing import Dict, List, Optional, Any, Union, Callable, Tuple, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import contextlib
import contextvars
from collections import defaultdict, deque

from backend.app.monitoring import logging

try:
    from opentelemetry import trace
    from opentelemetry.trace import (
        Span,
        SpanKind,
        Status,
        StatusCode,
        TracerProvider,
        set_tracer_provider,
    )
    from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )
    from opentelemetry.sdk.trace.sampling import (
        TraceIdRatioBased,
        ParentBased,
        AlwaysOnSampler,
        AlwaysOffSampler,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter # type: ignore
    from opentelemetry.exporter.zipkin.json import ZipkinExporter  # type: ignore
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.propagate import set_global_textmap, get_global_textmap
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    from opentelemetry.context import attach, detach, set_value
    from opentelemetry.semconv.trace import SpanAttributes
    
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

try:
    import opentracing
    from opentracing import Tracer as OpenTracingTracer
    from opentracing.ext import tags as ot_tags
    OPENTRACING_AVAILABLE = True
except ImportError:
    OPENTRACING_AVAILABLE = False

logger = logging.getLogger(__name__)


class TraceSampler(Enum):
    """Trace sampling strategies."""
    ALWAYS_ON = "always_on"
    ALWAYS_OFF = "always_off"
    PROBABILISTIC = "probabilistic"
    PARENT_BASED = "parent_based"
    RATE_LIMITING = "rate_limiting"


class TraceExporter(Enum):
    """Trace export destinations."""
    CONSOLE = "console"
    JAEGER = "jaeger"
    ZIPKIN = "zipkin"
    OTLP = "otlp"
    CUSTOM = "custom"


@dataclass
class TraceContext:
    """
    W3C Trace Context for distributed tracing.
    
    Contains trace ID, span ID, and other context information
    that needs to be propagated across service boundaries.
    """
    trace_id: str
    span_id: str
    trace_flags: str = "01"  # Sampled flag
    trace_state: Optional[str] = None
    is_remote: bool = False
    
    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP headers for propagation."""
        return {
            "traceparent": f"00-{self.trace_id}-{self.span_id}-{self.trace_flags}",
            "tracestate": self.trace_state or "",
        }
    
    @classmethod
    def from_headers(cls, headers: Dict[str, str]) -> Optional['TraceContext']:
        """Create TraceContext from HTTP headers."""
        traceparent = headers.get("traceparent")
        if not traceparent:
            return None
        
        try:
            # Parse traceparent: 00-trace_id-span_id-trace_flags
            parts = traceparent.split("-")
            if len(parts) != 4 or parts[0] != "00":
                return None
            
            return cls(
                trace_id=parts[1],
                span_id=parts[2],
                trace_flags=parts[3],
                trace_state=headers.get("tracestate"),
                is_remote=True,
            )
        except (ValueError, IndexError):
            return None
    
    @classmethod
    def generate(cls) -> 'TraceContext':
        """Generate a new trace context."""
        # Generate UUID-based IDs (OpenTelemetry compatible)
        trace_id = uuid.uuid4().hex[:32]  # 16 bytes hex
        span_id = uuid.uuid4().hex[:16]   # 8 bytes hex
        
        return cls(
            trace_id=trace_id,
            span_id=span_id,
            trace_flags="01",  # Sampled
        )


@dataclass
class SpanAttributes:
    """Common span attribute keys."""
    # HTTP
    HTTP_METHOD = "http.method"
    HTTP_URL = "http.url"
    HTTP_STATUS_CODE = "http.status_code"
    HTTP_ROUTE = "http.route"
    HTTP_CLIENT_IP = "http.client_ip"
    HTTP_USER_AGENT = "http.user_agent"
    
    # Database
    DB_SYSTEM = "db.system"
    DB_NAME = "db.name"
    DB_OPERATION = "db.operation"
    DB_STATEMENT = "db.statement"
    DB_USER = "db.user"
    
    # Message
    MESSAGING_SYSTEM = "messaging.system"
    MESSAGING_DESTINATION = "messaging.destination"
    MESSAGING_OPERATION = "messaging.operation"
    
    # RPC
    RPC_SYSTEM = "rpc.system"
    RPC_SERVICE = "rpc.service"
    RPC_METHOD = "rpc.method"
    
    # Custom
    SERVICE_NAME = "service.name"
    SERVICE_VERSION = "service.version"
    SERVICE_INSTANCE_ID = "service.instance.id"
    
    # Error
    ERROR_TYPE = "error.type"
    ERROR_MESSAGE = "error.message"
    ERROR_STACK_TRACE = "error.stack_trace"
    
    # Performance
    DURATION_MS = "duration.ms"
    RESPONSE_SIZE_BYTES = "response.size.bytes"
    REQUEST_SIZE_BYTES = "request.size.bytes"
    
    # Business
    USER_ID = "user.id"
    TENANT_ID = "tenant.id"
    ORGANIZATION_ID = "organization.id"
    PROJECT_ID = "project.id"
    OPERATION_NAME = "operation.name"
    
    @classmethod
    def get_all(cls) -> List[str]:
        """Get all attribute keys."""
        return [
            value for key, value in cls.__dict__.items()
            if not key.startswith('_') and isinstance(value, str)
        ]


@dataclass
class SpanMetadata:
    """Metadata for a span."""
    name: str
    kind: str = "INTERNAL"  # INTERNAL, SERVER, CLIENT, PRODUCER, CONSUMER
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    links: List[Dict[str, Any]] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    parent_span_id: Optional[str] = None
    status_code: str = "OK"
    status_description: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "name": self.name,
            "kind": self.kind,
            "attributes": self.attributes,
            "events": self.events,
            "links": self.links,
            "status_code": self.status_code,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
        }
        
        if self.start_time:
            result["start_time"] = self.start_time.isoformat()
        if self.end_time:
            result["end_time"] = self.end_time.isoformat()
        if self.status_description:
            result["status_description"] = self.status_description
        
        return result
    
    @property
    def duration_ms(self) -> Optional[float]:
        """Get span duration in milliseconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return None


class SpanRecorder:
    """
    Records span data for analysis and debugging.
    
    This can be used independently of OpenTelemetry for
    lightweight tracing or when OpenTelemetry is not available.
    """
    
    def __init__(self, max_spans: int = 10000):
        """
        Initialize span recorder.
        
        Args:
            max_spans: Maximum number of spans to keep in memory
        """
        self.max_spans = max_spans
        self.spans: Dict[str, SpanMetadata] = {}
        self.trace_map: Dict[str, List[str]] = defaultdict(list)  # trace_id -> span_ids
        self._lock = threading.RLock()
        
        # Active spans stack per thread
        self._active_spans = contextvars.ContextVar('active_spans', default=[])
    
    def start_span(
        self,
        name: str,
        kind: str = "INTERNAL",
        attributes: Optional[Dict[str, Any]] = None,
        trace_context: Optional[TraceContext] = None,
        parent_span_id: Optional[str] = None,
    ) -> Tuple[SpanMetadata, TraceContext]:
        """
        Start a new span.
        
        Args:
            name: Span name
            kind: Span kind
            attributes: Span attributes
            trace_context: Trace context (if None, create new)
            parent_span_id: Parent span ID
            
        Returns:
            Tuple[SpanMetadata, TraceContext]: Span metadata and trace context
        """
        # Generate trace context if not provided
        if trace_context is None:
            trace_context = TraceContext.generate()
        
        # Create span metadata
        span = SpanMetadata(
            name=name,
            kind=kind,
            attributes=attributes or {},
            start_time=datetime.utcnow(),
            trace_id=trace_context.trace_id,
            span_id=trace_context.span_id,
            parent_span_id=parent_span_id,
        )
        
        with self._lock:
            # Store span
            span_id = trace_context.span_id
            self.spans[span_id] = span
            
            # Update trace map
            self.trace_map[trace_context.trace_id].append(span_id)
            
            # Trim if needed
            if len(self.spans) > self.max_spans:
                self._trim_spans()
        
        # Add to active spans stack
        active_spans = self._active_spans.get()
        active_spans.append(span_id)
        self._active_spans.set(active_spans)
        
        return span, trace_context
    
    def end_span(
        self,
        span_id: str,
        status_code: str = "OK",
        status_description: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        events: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[SpanMetadata]:
        """
        End a span.
        
        Args:
            span_id: Span ID to end
            status_code: Span status code
            status_description: Status description
            attributes: Additional attributes to add
            events: Events to add
            
        Returns:
            Optional[SpanMetadata]: Ended span metadata
        """
        with self._lock:
            span = self.spans.get(span_id)
            if not span:
                return None
            
            # Update span
            span.end_time = datetime.utcnow()
            span.status_code = status_code
            span.status_description = status_description
            
            if attributes:
                span.attributes.update(attributes)
            
            if events:
                span.events.extend(events)
        
        # Remove from active spans stack
        active_spans = self._active_spans.get()
        if span_id in active_spans:
            active_spans.remove(span_id)
            self._active_spans.set(active_spans)
        
        return span
    
    def get_active_span_id(self) -> Optional[str]:
        """Get current active span ID."""
        active_spans = self._active_spans.get()
        return active_spans[-1] if active_spans else None
    
    def get_span(self, span_id: str) -> Optional[SpanMetadata]:
        """Get span by ID."""
        with self._lock:
            return self.spans.get(span_id)
    
    def get_trace(self, trace_id: str) -> List[SpanMetadata]:
        """Get all spans for a trace."""
        with self._lock:
            span_ids = self.trace_map.get(trace_id, [])
            return [self.spans[span_id] for span_id in span_ids if span_id in self.spans]
    
    def get_traces(
        self,
        limit: int = 100,
        min_duration_ms: Optional[float] = None,
        max_duration_ms: Optional[float] = None,
        service_name: Optional[str] = None,
        operation_name: Optional[str] = None,
        error_only: bool = False,
    ) -> Dict[str, List[SpanMetadata]]:
        """
        Get traces with filtering.
        
        Args:
            limit: Maximum number of traces to return
            min_duration_ms: Minimum trace duration in ms
            max_duration_ms: Maximum trace duration in ms
            service_name: Filter by service name
            operation_name: Filter by operation name
            error_only: Only return traces with errors
            
        Returns:
            Dict[str, List[SpanMetadata]]: Traces with their spans
        """
        traces = {}
        
        with self._lock:
            for trace_id, span_ids in self.trace_map.items():
                # Get all spans for this trace
                spans = [self.spans[span_id] for span_id in span_ids if span_id in self.spans]
                
                if not spans:
                    continue
                
                # Calculate trace duration
                start_time = min(s.start_time for s in spans if s.start_time)
                end_time = max(s.end_time for s in spans if s.end_time)
                
                if start_time and end_time:
                    duration_ms = (end_time - start_time).total_seconds() * 1000
                else:
                    duration_ms = 0.0
                
                # Apply filters
                if min_duration_ms is not None and duration_ms < min_duration_ms:
                    continue
                
                if max_duration_ms is not None and duration_ms > max_duration_ms:
                    continue
                
                if service_name:
                    if not any(s.attributes.get(SpanAttributes.SERVICE_NAME) == service_name for s in spans):
                        continue
                
                if operation_name:
                    if not any(s.attributes.get(SpanAttributes.OPERATION_NAME) == operation_name for s in spans):
                        continue
                
                if error_only:
                    if not any(s.status_code == "ERROR" for s in spans):
                        continue
                
                traces[trace_id] = spans
                
                # Limit results
                if len(traces) >= limit:
                    break
        
        return traces
    
    def _trim_spans(self) -> None:
        """Trim oldest spans to stay within limit."""
        # Simple LRU strategy: remove oldest spans
        with self._lock:
            # Get all span IDs sorted by end time (oldest first)
            span_ids_by_age = sorted(
                self.spans.keys(),
                key=lambda sid: self.spans[sid].end_time or datetime.min
            )
            
            # Remove excess spans
            excess = len(self.spans) - self.max_spans
            if excess > 0:
                for span_id in span_ids_by_age[:excess]:
                    # Remove from trace map
                    span = self.spans[span_id]
                    if span.trace_id in self.trace_map:
                        trace_span_ids = self.trace_map[span.trace_id]
                        if span_id in trace_span_ids:
                            trace_span_ids.remove(span_id)
                        
                        # Remove trace if empty
                        if not trace_span_ids:
                            del self.trace_map[span.trace_id]
                    
                    # Remove span
                    del self.spans[span_id]
    
    def clear(self) -> None:
        """Clear all spans."""
        with self._lock:
            self.spans.clear()
            self.trace_map.clear()
        
        # Clear active spans
        self._active_spans.set([])


class TracingManager:
    """
    Manages distributed tracing configuration and operations.
    
    Supports both OpenTelemetry and lightweight span recording.
    """
    
    def __init__(
        self,
        service_name: str = "worldbrief-360",
        service_version: str = "1.0.0",
        environment: str = "development",
        sampler: TraceSampler = TraceSampler.PROBABILISTIC,
        sample_rate: float = 0.1,  # 10% sampling
        exporters: List[TraceExporter] = None,
        enable_opentelemetry: bool = True,
        max_spans_in_memory: int = 10000,
        **kwargs
    ):
        """
        Initialize tracing manager.
        
        Args:
            service_name: Name of the service
            service_version: Service version
            environment: Deployment environment
            sampler: Trace sampling strategy
            sample_rate: Sampling rate for probabilistic sampling
            exporters: List of exporters to use
            enable_opentelemetry: Enable OpenTelemetry integration
            max_spans_in_memory: Maximum spans to keep in memory
            **kwargs: Additional configuration
        """
        self.service_name = service_name
        self.service_version = service_version
        self.environment = environment
        self.sampler = sampler
        self.sample_rate = sample_rate
        self.exporters = exporters or [TraceExporter.CONSOLE]
        self.enable_opentelemetry = enable_opentelemetry and OPENTELEMETRY_AVAILABLE
        self.max_spans_in_memory = max_spans_in_memory
        
        # Initialize components
        self.span_recorder = SpanRecorder(max_spans=max_spans_in_memory)
        
        # OpenTelemetry components
        self.tracer_provider = None
        self.tracer = None
        self.propagator = None
        
        if self.enable_opentelemetry:
            self._setup_opentelemetry()
        else:
            logger.warning("OpenTelemetry not available or disabled. Using lightweight tracing only.")
        
        # Context propagation
        self._current_context = contextvars.ContextVar('tracing_context', default=None)
        
        # Statistics
        self.stats = {
            "spans_started": 0,
            "spans_ended": 0,
            "traces_started": 0,
            "errors_recorded": 0,
            "last_span_time": None,
        }
        
        logger.info(f"TracingManager initialized for {service_name} in {environment}")
    
    def _setup_opentelemetry(self) -> None:
        """Setup OpenTelemetry tracing."""
        try:
            # Create resource
            resource = Resource.create({
                "service.name": self.service_name,
                "service.version": self.service_version,
                "deployment.environment": self.environment,
            })
            
            # Configure sampler
            sampler = self._create_sampler()
            
            # Create tracer provider
            self.tracer_provider = SDKTracerProvider(
                sampler=sampler,
                resource=resource,
            )
            
            # Add exporters
            for exporter_type in self.exporters:
                exporter = self._create_exporter(exporter_type)
                if exporter:
                    span_processor = BatchSpanProcessor(exporter)
                    self.tracer_provider.add_span_processor(span_processor)
            
            # Set global tracer provider
            set_tracer_provider(self.tracer_provider)
            
            # Get tracer
            self.tracer = trace.get_tracer(
                instrumenting_module_name=self.service_name,
                instrumenting_library_version=self.service_version,
            )
            
            # Setup propagator
            self.propagator = TraceContextTextMapPropagator()
            set_global_textmap(self.propagator)
            
            logger.info("OpenTelemetry tracing configured")
            
        except Exception as e:
            logger.error(f"Failed to setup OpenTelemetry: {e}")
            self.enable_opentelemetry = False
    
    def _create_sampler(self):
        """Create OpenTelemetry sampler based on configuration."""
        if self.sampler == TraceSampler.ALWAYS_ON:
            return AlwaysOnSampler()
        elif self.sampler == TraceSampler.ALWAYS_OFF:
            return AlwaysOffSampler()
        elif self.sampler == TraceSampler.PROBABILISTIC:
            return TraceIdRatioBased(self.sample_rate)
        elif self.sampler == TraceSampler.PARENT_BASED:
            return ParentBased(root=TraceIdRatioBased(self.sample_rate))
        else:
            return AlwaysOnSampler()
    
    def _create_exporter(self, exporter_type: TraceExporter):
        """Create OpenTelemetry exporter."""
        try:
            if exporter_type == TraceExporter.CONSOLE:
                return ConsoleSpanExporter()
            
            elif exporter_type == TraceExporter.JAEGER:
                return JaegerExporter(
                    agent_host_name="localhost",
                    agent_port=6831,
                )
            
            elif exporter_type == TraceExporter.ZIPKIN:
                return ZipkinExporter(
                    endpoint="http://localhost:9411/api/v2/spans",
                )
            
            elif exporter_type == TraceExporter.OTLP:
                return OTLPSpanExporter(
                    endpoint="localhost:4317",
                    insecure=True,
                )
            
            else:
                logger.warning(f"Unknown exporter type: {exporter_type}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to create exporter {exporter_type}: {e}")
            return None
    
    def start_trace(
        self,
        name: str,
        kind: str = "INTERNAL",
        attributes: Optional[Dict[str, Any]] = None,
        trace_context: Optional[TraceContext] = None,
        parent_context: Optional[Any] = None,
    ) -> Tuple[Any, TraceContext]:
        """
        Start a new trace/span.
        
        Args:
            name: Span name
            kind: Span kind
            attributes: Span attributes
            trace_context: Existing trace context
            parent_context: Parent context (for OpenTelemetry)
            
        Returns:
            Tuple[Any, TraceContext]: Span context and trace context
        """
        # Prepare attributes
        all_attributes = {
            SpanAttributes.SERVICE_NAME: self.service_name,
            SpanAttributes.SERVICE_VERSION: self.service_version,
            **(attributes or {}),
        }
        
        # Get parent span ID for recorder
        parent_span_id = None
        if parent_context and hasattr(parent_context, 'span_id'):
            parent_span_id = parent_context.span_id
        
        # Start span in recorder
        span_metadata, trace_context = self.span_recorder.start_span(
            name=name,
            kind=kind,
            attributes=all_attributes,
            trace_context=trace_context,
            parent_span_id=parent_span_id,
        )
        
        # Start OpenTelemetry span if enabled
        otel_span = None
        if self.enable_opentelemetry and self.tracer:
            try:
                # Map span kind
                span_kind = self._map_span_kind(kind)
                
                # Create span
                otel_span = self.tracer.start_span(
                    name=name,
                    kind=span_kind,
                    attributes=all_attributes,
                    start_time=int(span_metadata.start_time.timestamp() * 1e9) if span_metadata.start_time else None,
                )
                
                # Set trace context
                if trace_context:
                    # Create OpenTelemetry context
                    from opentelemetry.trace import set_span_in_context
                    from opentelemetry.trace.span import INVALID_SPAN_ID, INVALID_TRACE_ID
                    
                    # Create span context
                    span_context = trace.SpanContext(
                        trace_id=int(trace_context.trace_id, 16),
                        span_id=int(trace_context.span_id, 16),
                        is_remote=trace_context.is_remote,
                        trace_flags=trace.TraceFlags(int(trace_context.trace_flags, 16)),
                        trace_state=trace.TraceState([(k, v) for k, v in (trace_context.trace_state or "").split(',') if k and v]),
                    )
                    
                    # Create span and set in context
                    otel_span._context = span_context
                
            except Exception as e:
                logger.error(f"Failed to start OpenTelemetry span: {e}")
        
        # Update statistics
        self.stats["spans_started"] += 1
        self.stats["last_span_time"] = datetime.utcnow()
        
        if parent_span_id is None:
            self.stats["traces_started"] += 1
        
        # Store context
        context = {
            "recorder_span_id": span_metadata.span_id,
            "otel_span": otel_span,
            "trace_context": trace_context,
            "start_time": span_metadata.start_time,
        }
        
        self._current_context.set(context)
        
        return context, trace_context
    
    def _map_span_kind(self, kind: str) -> SpanKind:
        """Map span kind string to OpenTelemetry SpanKind."""
        kind_map = {
            "INTERNAL": SpanKind.INTERNAL,
            "SERVER": SpanKind.SERVER,
            "CLIENT": SpanKind.CLIENT,
            "PRODUCER": SpanKind.PRODUCER,
            "CONSUMER": SpanKind.CONSUMER,
        }
        return kind_map.get(kind.upper(), SpanKind.INTERNAL)
    
    def end_trace(
        self,
        context: Any,
        status_code: str = "OK",
        status_description: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        events: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[SpanMetadata]:
        """
        End a trace/span.
        
        Args:
            context: Span context from start_trace
            status_code: Span status code
            status_description: Status description
            attributes: Additional attributes
            events: Events to add
            
        Returns:
            Optional[SpanMetadata]: Ended span metadata
        """
        if not context:
            return None
        
        # End span in recorder
        span_metadata = None
        if "recorder_span_id" in context:
            span_metadata = self.span_recorder.end_span(
                span_id=context["recorder_span_id"],
                status_code=status_code,
                status_description=status_description,
                attributes=attributes,
                events=events,
            )
        
        # End OpenTelemetry span
        if self.enable_opentelemetry and "otel_span" in context and context["otel_span"]:
            try:
                otel_span = context["otel_span"]
                
                # Set status
                if status_code == "ERROR":
                    otel_span.set_status(Status(StatusCode.ERROR, status_description))
                else:
                    otel_span.set_status(Status(StatusCode.OK))
                
                # Add attributes
                if attributes:
                    for key, value in attributes.items():
                        otel_span.set_attribute(key, value)
                
                # Add events
                if events:
                    for event in events:
                        otel_span.add_event(
                            event.get("name", "event"),
                            attributes=event.get("attributes", {}),
                            timestamp=event.get("timestamp"),
                        )
                
                # End span
                otel_span.end()
                
            except Exception as e:
                logger.error(f"Failed to end OpenTelemetry span: {e}")
        
        # Update statistics
        self.stats["spans_ended"] += 1
        if status_code == "ERROR":
            self.stats["errors_recorded"] += 1
        
        # Clear current context if this was the active span
        current_context = self._current_context.get()
        if current_context and current_context.get("recorder_span_id") == context.get("recorder_span_id"):
            self._current_context.set(None)
        
        return span_metadata
    
    def get_current_context(self) -> Optional[Dict[str, Any]]:
        """Get current tracing context."""
        return self._current_context.get()
    
    def get_current_trace_context(self) -> Optional[TraceContext]:
        """Get current trace context."""
        context = self.get_current_context()
        if context and "trace_context" in context:
            return context["trace_context"]
        return None
    
    def inject_trace_context(self, carrier: Dict[str, str]) -> None:
        """
        Inject trace context into carrier for propagation.
        
        Args:
            carrier: Dictionary to inject trace context into
        """
        trace_context = self.get_current_trace_context()
        if trace_context:
            carrier.update(trace_context.to_headers())
        
        # Also inject using OpenTelemetry propagator
        if self.enable_opentelemetry and self.propagator:
            try:
                from opentelemetry.trace import get_current_span
                from opentelemetry.context import get_current
                
                current_span = get_current_span()
                if current_span:
                    ctx = trace.set_span_in_context(current_span)
                    self.propagator.inject(carrier, context=ctx)
            except Exception as e:
                logger.error(f"Failed to inject OpenTelemetry context: {e}")
    
    def extract_trace_context(self, carrier: Dict[str, str]) -> Optional[TraceContext]:
        """
        Extract trace context from carrier.
        
        Args:
            carrier: Dictionary containing trace context
            
        Returns:
            Optional[TraceContext]: Extracted trace context
        """
        # Try OpenTelemetry extraction first
        if self.enable_opentelemetry and self.propagator:
            try:
                ctx = self.propagator.extract(carrier)
                if ctx:
                    from opentelemetry.trace import get_current_span
                    span = get_current_span(ctx)
                    if span:
                        span_context = span.get_span_context()
                        if span_context.is_valid:
                            return TraceContext(
                                trace_id=format(span_context.trace_id, '032x'),
                                span_id=format(span_context.span_id, '016x'),
                                trace_flags=f"{span_context.trace_flags:02x}",
                                trace_state=str(span_context.trace_state),
                                is_remote=True,
                            )
            except Exception as e:
                logger.error(f"Failed to extract OpenTelemetry context: {e}")
        
        # Fallback to W3C trace context extraction
        return TraceContext.from_headers(carrier)
    
    @contextlib.contextmanager
    def trace(
        self,
        name: str,
        kind: str = "INTERNAL",
        attributes: Optional[Dict[str, Any]] = None,
        trace_context: Optional[TraceContext] = None,
    ):
        """
        Context manager for tracing.
        
        Args:
            name: Span name
            kind: Span kind
            attributes: Span attributes
            trace_context: Existing trace context
        """
        context, _ = self.start_trace(
            name=name,
            kind=kind,
            attributes=attributes,
            trace_context=trace_context,
        )
        
        try:
            yield context
            self.end_trace(context, status_code="OK")
        except Exception as e:
            self.end_trace(
                context,
                status_code="ERROR",
                status_description=str(e),
                attributes={"error": str(e), "error_type": type(e).__name__},
            )
            raise
    
    @contextlib.asynccontextmanager
    async def trace_async(
        self,
        name: str,
        kind: str = "INTERNAL",
        attributes: Optional[Dict[str, Any]] = None,
        trace_context: Optional[TraceContext] = None,
    ):
        """
        Async context manager for tracing.
        
        Args:
            name: Span name
            kind: Span kind
            attributes: Span attributes
            trace_context: Existing trace context
        """
        context, _ = self.start_trace(
            name=name,
            kind=kind,
            attributes=attributes,
            trace_context=trace_context,
        )
        
        try:
            yield context
            self.end_trace(context, status_code="OK")
        except Exception as e:
            self.end_trace(
                context,
                status_code="ERROR",
                status_description=str(e),
                attributes={"error": str(e), "error_type": type(e).__name__},
            )
            raise
    
    def record_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Record an event on the current span.
        
        Args:
            name: Event name
            attributes: Event attributes
            timestamp: Event timestamp
        """
        context = self.get_current_context()
        if not context:
            return
        
        event = {
            "name": name,
            "attributes": attributes or {},
            "timestamp": timestamp or datetime.utcnow(),
        }
        
        # Record in span recorder
        if "recorder_span_id" in context:
            span = self.span_recorder.get_span(context["recorder_span_id"])
            if span:
                span.events.append(event)
        
        # Record in OpenTelemetry span
        if self.enable_opentelemetry and "otel_span" in context and context["otel_span"]:
            try:
                otel_span = context["otel_span"]
                otel_span.add_event(
                    name=name,
                    attributes=attributes or {},
                    timestamp=int(timestamp.timestamp() * 1e9) if timestamp else None,
                )
            except Exception as e:
                logger.error(f"Failed to record OpenTelemetry event: {e}")
    
    def set_attribute(self, key: str, value: Any) -> None:
        """
        Set attribute on current span.
        
        Args:
            key: Attribute key
            value: Attribute value
        """
        context = self.get_current_context()
        if not context:
            return
        
        # Set in span recorder
        if "recorder_span_id" in context:
            span = self.span_recorder.get_span(context["recorder_span_id"])
            if span:
                span.attributes[key] = value
        
        # Set in OpenTelemetry span
        if self.enable_opentelemetry and "otel_span" in context and context["otel_span"]:
            try:
                otel_span = context["otel_span"]
                otel_span.set_attribute(key, value)
            except Exception as e:
                logger.error(f"Failed to set OpenTelemetry attribute: {e}")
    
    def get_traces(
        self,
        limit: int = 100,
        min_duration_ms: Optional[float] = None,
        max_duration_ms: Optional[float] = None,
        service_name: Optional[str] = None,
        operation_name: Optional[str] = None,
        error_only: bool = False,
    ) -> Dict[str, List[SpanMetadata]]:
        """
        Get traces with filtering.
        
        Args:
            limit: Maximum number of traces to return
            min_duration_ms: Minimum trace duration in ms
            max_duration_ms: Maximum trace duration in ms
            service_name: Filter by service name
            operation_name: Filter by operation name
            error_only: Only return traces with errors
            
        Returns:
            Dict[str, List[SpanMetadata]]: Traces with their spans
        """
        return self.span_recorder.get_traces(
            limit=limit,
            min_duration_ms=min_duration_ms,
            max_duration_ms=max_duration_ms,
            service_name=service_name,
            operation_name=operation_name,
            error_only=error_only,
        )
    
    def get_trace(self, trace_id: str) -> List[SpanMetadata]:
        """Get all spans for a trace."""
        return self.span_recorder.get_trace(trace_id)
    
    def get_span(self, span_id: str) -> Optional[SpanMetadata]:
        """Get span by ID."""
        return self.span_recorder.get_span(span_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get tracing statistics.
        
        Returns:
            Dict[str, Any]: Statistics dictionary
        """
        stats = self.stats.copy()
        
        # Add span recorder stats
        with self.span_recorder._lock:
            stats["spans_in_memory"] = len(self.span_recorder.spans)
            stats["traces_in_memory"] = len(self.span_recorder.trace_map)
        
        # Add configuration
        stats["configuration"] = {
            "service_name": self.service_name,
            "service_version": self.service_version,
            "environment": self.environment,
            "sampler": self.sampler.value,
            "sample_rate": self.sample_rate,
            "exporters": [e.value for e in self.exporters],
            "opentelemetry_enabled": self.enable_opentelemetry,
            "max_spans_in_memory": self.max_spans_in_memory,
        }
        
        return stats
    
    def clear_spans(self) -> None:
        """Clear all spans from memory."""
        self.span_recorder.clear()
        logger.info("Cleared all spans from memory")
    
    def shutdown(self) -> None:
        """Shutdown tracing manager."""
        if self.tracer_provider:
            self.tracer_provider.shutdown()
        
        self.clear_spans()
        logger.info("TracingManager shut down")


# Global tracing manager instance
_tracing_manager: Optional[TracingManager] = None


def setup_tracing(
    service_name: str = "worldbrief-360",
    service_version: str = "1.0.0",
    environment: str = "development",
    sampler: Union[str, TraceSampler] = "probabilistic",
    sample_rate: float = 0.1,
    exporters: List[Union[str, TraceExporter]] = None,
    enable_opentelemetry: bool = True,
    **kwargs
) -> TracingManager:
    """
    Set up distributed tracing.
    
    Args:
        service_name: Name of the service
        service_version: Service version
        environment: Deployment environment
        sampler: Trace sampling strategy
        sample_rate: Sampling rate for probabilistic sampling
        exporters: List of exporters to use
        enable_opentelemetry: Enable OpenTelemetry integration
        **kwargs: Additional configuration
        
    Returns:
        TracingManager: Configured tracing manager
    """
    global _tracing_manager
    
    if _tracing_manager is not None:
        logger.warning("Tracing already set up. Returning existing instance.")
        return _tracing_manager
    
    # Parse sampler
    if isinstance(sampler, str):
        try:
            sampler = TraceSampler(sampler.lower())
        except ValueError:
            logger.warning(f"Unknown sampler: {sampler}. Using probabilistic.")
            sampler = TraceSampler.PROBABILISTIC
    
    # Parse exporters
    parsed_exporters = []
    if exporters:
        for exporter in exporters:
            if isinstance(exporter, str):
                try:
                    parsed_exporters.append(TraceExporter(exporter.lower()))
                except ValueError:
                    logger.warning(f"Unknown exporter: {exporter}. Skipping.")
            else:
                parsed_exporters.append(exporter)
    else:
        parsed_exporters = [TraceExporter.CONSOLE]
    
    _tracing_manager = TracingManager(
        service_name=service_name,
        service_version=service_version,
        environment=environment,
        sampler=sampler,
        sample_rate=sample_rate,
        exporters=parsed_exporters,
        enable_opentelemetry=enable_opentelemetry,
        **kwargs
    )
    
    logger.info(f"Tracing set up for {service_name} (v{service_version}) in {environment}")
    
    return _tracing_manager


def get_tracer() -> TracingManager:
    """
    Get the global tracing manager instance.
    
    Returns:
        TracingManager: Global tracing manager
        
    Raises:
        RuntimeError: If tracing is not set up
    """
    if _tracing_manager is None:
        raise RuntimeError("Tracing not set up. Call setup_tracing() first.")
    
    return _tracing_manager


# Decorators for easy tracing
def trace_function(
    name: Optional[str] = None,
    kind: str = "INTERNAL",
    attributes: Optional[Dict[str, Any]] = None,
):
    """
    Decorator to trace function execution.
    
    Args:
        name: Span name (defaults to function name)
        kind: Span kind
        attributes: Span attributes
    """
    def decorator(func):
        span_name = name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            
            # Prepare span attributes
            all_attributes = (attributes or {}).copy()
            all_attributes.update({
                "function": func.__name__,
                "module": func.__module__,
                "args_count": len(args),
                "kwargs_count": len(kwargs),
            })
            
            # Extract trace context from kwargs if present
            trace_context = None
            if "trace_context" in kwargs:
                trace_context = kwargs.pop("trace_context")
            
            with tracer.trace(
                name=span_name,
                kind=kind,
                attributes=all_attributes,
                trace_context=trace_context,
            ) as span_context:
                # Add args/kwargs as attributes (careful with sensitive data)
                try:
                    if args:
                        tracer.set_attribute("args.types", [type(arg).__name__ for arg in args])
                    if kwargs:
                        tracer.set_attribute("kwargs.keys", list(kwargs.keys()))
                except:
                    pass
                
                # Execute function
                return func(*args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = get_tracer()
            
            # Prepare span attributes
            all_attributes = (attributes or {}).copy()
            all_attributes.update({
                "function": func.__name__,
                "module": func.__module__,
                "args_count": len(args),
                "kwargs_count": len(kwargs),
            })
            
            # Extract trace context from kwargs if present
            trace_context = None
            if "trace_context" in kwargs:
                trace_context = kwargs.pop("trace_context")
            
            async with tracer.trace_async(
                name=span_name,
                kind=kind,
                attributes=all_attributes,
                trace_context=trace_context,
            ) as span_context:
                # Add args/kwargs as attributes
                try:
                    if args:
                        tracer.set_attribute("args.types", [type(arg).__name__ for arg in args])
                    if kwargs:
                        tracer.set_attribute("kwargs.keys", list(kwargs.keys()))
                except:
                    pass
                
                # Execute async function
                return await func(*args, **kwargs)
        
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


def trace_coroutine(
    name: Optional[str] = None,
    kind: str = "INTERNAL",
    attributes: Optional[Dict[str, Any]] = None,
):
    """
    Decorator specifically for coroutines.
    
    Args:
        name: Span name
        kind: Span kind
        attributes: Span attributes
    """
    def decorator(coro):
        span_name = name or coro.__name__
        
        @functools.wraps(coro)
        async def wrapper(*args, **kwargs):
            tracer = get_tracer()
            
            all_attributes = (attributes or {}).copy()
            all_attributes.update({
                "coroutine": coro.__name__,
                "module": coro.__module__,
            })
            
            async with tracer.trace_async(
                name=span_name,
                kind=kind,
                attributes=all_attributes,
            ):
                return await coro(*args, **kwargs)
        
        return wrapper
    
    return decorator


def trace_http_request(
    endpoint: str,
    method: str = "GET",
    attributes: Optional[Dict[str, Any]] = None,
):
    """
    Decorator for tracing HTTP requests.
    
    Args:
        endpoint: HTTP endpoint
        method: HTTP method
        attributes: Additional attributes
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            tracer = get_tracer()
            
            # Extract request object if available
            request = None
            for arg in args:
                if hasattr(arg, 'method') and hasattr(arg, 'url'):
                    request = arg
                    break
            
            # Prepare attributes
            all_attributes = {
                SpanAttributes.HTTP_METHOD: method,
                SpanAttributes.HTTP_ROUTE: endpoint,
                SpanAttributes.OPERATION_NAME: f"{method} {endpoint}",
                **(attributes or {}),
            }
            
            # Add request-specific attributes
            if request:
                all_attributes.update({
                    SpanAttributes.HTTP_URL: str(request.url),
                    SpanAttributes.HTTP_CLIENT_IP: request.client.host if request.client else None,
                    SpanAttributes.HTTP_USER_AGENT: request.headers.get("user-agent"),
                })
                
                # Extract trace context from headers
                headers = dict(request.headers)
                trace_context = tracer.extract_trace_context(headers)
            else:
                trace_context = None
            
            # Start trace
            async with tracer.trace_async(
                name=f"{method} {endpoint}",
                kind="SERVER",
                attributes=all_attributes,
                trace_context=trace_context,
            ) as span_context:
                # Inject trace context into response
                try:
                    response = await func(*args, **kwargs)
                    
                    # Add status code attribute
                    if hasattr(response, 'status_code'):
                        tracer.set_attribute(SpanAttributes.HTTP_STATUS_CODE, response.status_code)
                    
                    return response
                    
                except Exception as e:
                    tracer.set_attribute(SpanAttributes.ERROR_TYPE, type(e).__name__)
                    tracer.set_attribute(SpanAttributes.ERROR_MESSAGE, str(e))
                    raise
        
        return wrapper
    
    return decorator


def get_trace_context() -> Optional[TraceContext]:
    """
    Get current trace context.
    
    Returns:
        Optional[TraceContext]: Current trace context
    """
    try:
        tracer = get_tracer()
        return tracer.get_current_trace_context()
    except RuntimeError:
        return None


def inject_trace_context(carrier: Dict[str, str]) -> None:
    """
    Inject trace context into carrier.
    
    Args:
        carrier: Dictionary to inject trace context into
    """
    try:
        tracer = get_tracer()
        tracer.inject_trace_context(carrier)
    except RuntimeError:
        pass


def extract_trace_context(carrier: Dict[str, str]) -> Optional[TraceContext]:
    """
    Extract trace context from carrier.
    
    Args:
        carrier: Dictionary containing trace context
        
    Returns:
        Optional[TraceContext]: Extracted trace context
    """
    try:
        tracer = get_tracer()
        return tracer.extract_trace_context(carrier)
    except RuntimeError:
        return TraceContext.from_headers(carrier)


# Example usage
if __name__ == "__main__":
    import asyncio
    
    # Setup tracing
    tracer = setup_tracing(
        service_name="test-service",
        service_version="1.0.0",
        environment="development",
        sampler="probabilistic",
        sample_rate=0.5,
        exporters=["console"],
        enable_opentelemetry=OPENTELEMETRY_AVAILABLE,
    )
    
    # Test tracing with decorators
    @trace_function(name="test_function", kind="INTERNAL")
    def test_function(x, y):
        time.sleep(0.1)
        return x + y
    
    @trace_coroutine(name="test_coroutine")
    async def test_coroutine():
        await asyncio.sleep(0.05)
        return "coroutine result"
    
    # Test HTTP tracing
    @trace_http_request(endpoint="/api/test", method="GET")
    async def test_http_endpoint():
        return {"status": "ok"}
    
    # Run tests
    print("Testing distributed tracing...")
    
    # Test sync function
    result = test_function(10, 20)
    print(f"Function result: {result}")
    
    # Test async functions
    async def run_tests():
        # Test coroutine
        coro_result = await test_coroutine()
        print(f"Coroutine result: {coro_result}")
        
        # Test HTTP endpoint
        http_result = await test_http_endpoint()
        print(f"HTTP endpoint result: {http_result}")
        
        # Test manual tracing
        with tracer.trace("manual_operation", kind="INTERNAL") as span_ctx:
            tracer.set_attribute("manual_attr", "value")
            tracer.record_event("manual_event", {"data": "test"})
            print("Manual trace completed")
        
        # Get trace context
        trace_ctx = get_trace_context()
        if trace_ctx:
            print(f"Current trace ID: {trace_ctx.trace_id}")
        
        # Test context propagation
        carrier = {}
        inject_trace_context(carrier)
        print(f"Injected trace context: {carrier}")
        
        # Get statistics
        stats = tracer.get_stats()
        print(f"\nTracing statistics: {stats}")
        
        # Get traces
        traces = tracer.get_traces(limit=5)
        print(f"\nCollected {len(traces)} traces")
        
        for trace_id, spans in traces.items():
            print(f"\nTrace {trace_id}:")
            for span in spans:
                print(f"  - {span.name}: {span.duration_ms:.2f}ms")
    
    asyncio.run(run_tests())
    
    # Cleanup
    tracer.shutdown()
    
    print("\nTracing test completed")