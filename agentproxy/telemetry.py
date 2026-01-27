"""
OpenTelemetry instrumentation for agentproxy.
Provides traces, metrics, and logs for PA operations.

This module has conditional imports - OTEL is optional.
If OTEL packages are not installed, telemetry is gracefully disabled.
"""

import os
from typing import Optional
import socket

# Try to import OTEL packages - they're optional
try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    # Stub classes for when OTEL is not available
    trace = None
    metrics = None


class NoOpTelemetry:
    """No-op telemetry when OTEL is not available or disabled."""

    def __init__(self):
        self.enabled = False
        self.tracer = None
        self.meter = None

    def instrument_fastapi(self, app):
        """No-op FastAPI instrumentation."""
        pass


if OTEL_AVAILABLE:
    class AgentProxyTelemetry:
        """Manages OTEL instrumentation for agentproxy"""

        def __init__(self):
            self.enabled = os.getenv("AGENTPROXY_ENABLE_TELEMETRY", "0") == "1"
            self.tracer: Optional[trace.Tracer] = None
            self.meter: Optional[metrics.Meter] = None

            if self.enabled:
                self._init_telemetry()

        def _init_telemetry(self):
            """Initialize OTEL providers and exporters"""
            # Build resource attributes
            resource = Resource.create({
                "service.name": os.getenv("OTEL_SERVICE_NAME", "agentproxy"),
                "service.namespace": os.getenv("OTEL_SERVICE_NAMESPACE", "default"),
                "host.name": os.getenv("HOSTNAME", socket.gethostname()),
                "agentproxy.owner": os.getenv("AGENTPROXY_OWNER_ID", os.getenv("USER", "unknown")),
            })

            # Traces
            trace_provider = TracerProvider(resource=resource)
            otlp_trace_exporter = OTLPSpanExporter(
                endpoint=os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or
                         os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
            )
            trace_provider.add_span_processor(BatchSpanProcessor(otlp_trace_exporter))
            trace.set_tracer_provider(trace_provider)
            self.tracer = trace.get_tracer(__name__)

            # Metrics
            otlp_metric_exporter = OTLPMetricExporter(
                endpoint=os.getenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT") or
                         os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
            )
            metric_reader = PeriodicExportingMetricReader(
                otlp_metric_exporter,
                export_interval_millis=int(os.getenv("OTEL_METRIC_EXPORT_INTERVAL", "10000")),
            )
            meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
            metrics.set_meter_provider(meter_provider)
            self.meter = metrics.get_meter(__name__)

            # Initialize metrics
            self._init_metrics()

        def _init_metrics(self):
            """Create metric instruments"""
            # Counters
            self.tasks_started = self.meter.create_counter(
                "agentproxy.tasks.started",
                description="Number of tasks started",
                unit="1",
            )
            self.tasks_completed = self.meter.create_counter(
                "agentproxy.tasks.completed",
                description="Number of tasks completed",
                unit="1",
            )
            self.claude_iterations = self.meter.create_counter(
                "agentproxy.claude.iterations",
                description="Number of Claude invocations",
                unit="1",
            )
            self.verifications = self.meter.create_counter(
                "agentproxy.verifications",
                description="Number of verifications run",
                unit="1",
            )
            self.pa_decisions = self.meter.create_counter(
                "agentproxy.pa.decisions",
                description="PA decisions made",
                unit="1",
            )

            # Histograms
            self.task_duration = self.meter.create_histogram(
                "agentproxy.task.duration",
                description="Task duration",
                unit="s",
            )
            self.pa_reasoning_duration = self.meter.create_histogram(
                "agentproxy.pa.reasoning.duration",
                description="PA reasoning cycle duration",
                unit="s",
            )
            self.gemini_api_duration = self.meter.create_histogram(
                "agentproxy.gemini.api.duration",
                description="Gemini API call duration",
                unit="s",
            )

            # Gauges
            self.active_sessions = self.meter.create_up_down_counter(
                "agentproxy.sessions.active",
                description="Number of active sessions",
                unit="1",
            )

        def instrument_fastapi(self, app):
            """Auto-instrument FastAPI if telemetry enabled"""
            if self.enabled:
                FastAPIInstrumentor.instrument_app(app)


# Global singleton
_telemetry: Optional[AgentProxyTelemetry] = None


def get_telemetry():
    """Get or create global telemetry instance"""
    global _telemetry

    if not OTEL_AVAILABLE:
        # OTEL not installed, return no-op
        if _telemetry is None:
            _telemetry = NoOpTelemetry()
        return _telemetry

    if _telemetry is None:
        _telemetry = AgentProxyTelemetry()
    return _telemetry
