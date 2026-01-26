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


# API Pricing (USD per 1M tokens, as of Jan 2025)
API_PRICING = {
    "gemini": {
        "gemini-2.5-flash": {"prompt": 0.075, "completion": 0.30},
        "gemini-2.0-flash": {"prompt": 0.075, "completion": 0.30},
    },
    "claude": {
        "claude-opus-4-5": {"prompt": 15.0, "completion": 75.0, "cache_write": 18.75, "cache_read": 1.50},
        "claude-sonnet-4-5": {"prompt": 3.0, "completion": 15.0, "cache_write": 3.75, "cache_read": 0.30},
    }
}

# Model context window limits
MODEL_CONTEXT_LIMITS = {
    "gemini-2.5-flash": 1_000_000,
    "gemini-2.0-flash": 1_000_000,
    "claude-opus-4-5": 200_000,
    "claude-sonnet-4-5": 200_000,
}


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
            self.verbose = os.getenv("AGENTPROXY_TELEMETRY_VERBOSE", "0") == "1"
            self.tracer: Optional[trace.Tracer] = None
            self.meter: Optional[metrics.Meter] = None

            # Print telemetry status
            if self.enabled:
                print("\033[2m│ 📊 OTEL     │\033[0m Telemetry ENABLED (AGENTPROXY_ENABLE_TELEMETRY=1)")
                if self.verbose:
                    print("\033[2m│ 📊 OTEL     │\033[0m Verbose logging ON (AGENTPROXY_TELEMETRY_VERBOSE=1)")
                self._init_telemetry()
            else:
                print("\033[2m│ 📊 OTEL     │\033[0m Telemetry disabled (set AGENTPROXY_ENABLE_TELEMETRY=1 to enable)")

        def _init_telemetry(self):
            """Initialize OTEL providers and exporters"""
            try:
                # Build resource attributes
                # Multi-tenant namespace: {user}.{project} for aggregation
                user_id = os.getenv("AGENTPROXY_OWNER_ID", os.getenv("USER", "unknown"))
                project_id = os.getenv("AGENTPROXY_PROJECT_ID", "default")
                namespace = os.getenv("OTEL_SERVICE_NAMESPACE", f"{user_id}.{project_id}")

                print(f"\033[2m│ 📊 OTEL     │\033[0m Initializing telemetry...")
                print(f"\033[2m│ 📊 OTEL     │\033[0m   Service: {os.getenv('OTEL_SERVICE_NAME', 'agentproxy')}")
                print(f"\033[2m│ 📊 OTEL     │\033[0m   Namespace: {namespace}")
                print(f"\033[2m│ 📊 OTEL     │\033[0m   Owner: {user_id}")
                print(f"\033[2m│ 📊 OTEL     │\033[0m   Project: {project_id}")
                print(f"\033[2m│ 📊 OTEL     │\033[0m   Role: {os.getenv('AGENTPROXY_ROLE', 'supervisor')}")

                resource = Resource.create({
                    "service.name": os.getenv("OTEL_SERVICE_NAME", "agentproxy"),
                    "service.namespace": namespace,
                    "host.name": os.getenv("HOSTNAME", socket.gethostname()),
                    "agentproxy.owner": user_id,
                    "agentproxy.project_id": project_id,
                    "agentproxy.role": os.getenv("AGENTPROXY_ROLE", "supervisor"),
                })

                # Traces
                trace_endpoint = (os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or
                                os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"))
                print(f"\033[2m│ 📊 OTEL     │\033[0m   Trace endpoint: {trace_endpoint}")

                # Safely parse trace export interval with fallback to default (1000ms = 1 second)
                # Lower values = more real-time visibility in Grafana but higher overhead
                # 1000ms provides good balance between latency and performance
                try:
                    trace_export_interval = int(os.getenv("OTEL_TRACE_EXPORT_INTERVAL", "1000"))
                except (ValueError, TypeError):
                    trace_export_interval = 1000

                print(f"\033[2m│ 📊 OTEL     │\033[0m   Trace export interval: {trace_export_interval}ms")

                trace_provider = TracerProvider(resource=resource)
                # Configure TLS based on environment (insecure for localhost, secure for production)
                use_insecure = os.getenv("OTEL_EXPORTER_OTLP_INSECURE", "true").lower() == "true"
                otlp_trace_exporter = OTLPSpanExporter(
                    endpoint=trace_endpoint,
                    insecure=use_insecure
                )
                print(f"\033[2m│ 📊 OTEL     │\033[0m   TLS: {'disabled (insecure)' if use_insecure else 'enabled (secure)'}")
                # Configure batch processor for periodic exports
                # schedule_delay_millis: time between exports (configured via env var)
                # max_queue_size: maximum spans to buffer before forced export
                # max_export_batch_size: maximum spans per export batch
                batch_processor = BatchSpanProcessor(
                    otlp_trace_exporter,
                    schedule_delay_millis=trace_export_interval,
                    max_queue_size=2048,  # Default is 2048
                    max_export_batch_size=512,  # Default is 512
                )
                trace_provider.add_span_processor(batch_processor)
                trace.set_tracer_provider(trace_provider)
                self.tracer = trace.get_tracer(__name__)

                # Metrics
                metric_endpoint = (os.getenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT") or
                                 os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"))
                print(f"\033[2m│ 📊 OTEL     │\033[0m   Metric endpoint: {metric_endpoint}")

                # Safely parse export interval with fallback to default
                try:
                    export_interval = int(os.getenv("OTEL_METRIC_EXPORT_INTERVAL", "10000"))
                except (ValueError, TypeError):
                    export_interval = 10000

                print(f"\033[2m│ 📊 OTEL     │\033[0m   Metric export interval: {export_interval}ms")

                # Configure TLS based on environment (same setting for traces and metrics)
                otlp_metric_exporter = OTLPMetricExporter(
                    endpoint=metric_endpoint,
                    insecure=use_insecure
                )
                metric_reader = PeriodicExportingMetricReader(
                    otlp_metric_exporter,
                    export_interval_millis=export_interval,
                )
                meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
                metrics.set_meter_provider(meter_provider)
                self.meter = metrics.get_meter(__name__)

                # Initialize metrics
                self._init_metrics()
                print(f"\033[2m│ 📊 OTEL     │\033[0m Telemetry initialization complete ✓")

            except Exception as e:
                print(f"\033[2m│ 📊 OTEL     │\033[0m \033[91mFailed to initialize telemetry: {e}\033[0m")
                print(f"\033[2m│ 📊 OTEL     │\033[0m Continuing without telemetry...")
                self.enabled = False
                self.verbose = False
                self.tracer = None
                self.meter = None

        def log(self, message: str) -> None:
            """Log telemetry message if verbose enabled."""
            if self.verbose:
                print(f"\033[2m│ 📊 OTEL     │\033[0m {message}")

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
            self.tokens_consumed = self.meter.create_counter(
                "agentproxy.tokens.consumed",
                description="Total LLM tokens consumed (prompt + completion)",
                unit="tokens",
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

            # Tool execution tracking
            self.tool_executions = self.meter.create_counter(
                "agentproxy.tools.executions",
                description="Tool executions by name and outcome",
                unit="1",
            )
            self.tool_duration = self.meter.create_histogram(
                "agentproxy.tools.duration",
                description="Tool execution duration by type",
                unit="s",
            )

            # Token breakdown (replace single tokens_consumed)
            self.tokens_prompt = self.meter.create_counter(
                "agentproxy.tokens.prompt",
                description="Prompt tokens by API and model",
                unit="tokens",
            )
            self.tokens_completion = self.meter.create_counter(
                "agentproxy.tokens.completion",
                description="Completion tokens by API and model",
                unit="tokens",
            )
            self.tokens_cache_write = self.meter.create_counter(
                "agentproxy.tokens.cache_write",
                description="Cache write tokens (future-ready)",
                unit="tokens",
            )
            self.tokens_cache_read = self.meter.create_counter(
                "agentproxy.tokens.cache_read",
                description="Cache read tokens (future-ready)",
                unit="tokens",
            )

            # API tracking
            self.api_requests = self.meter.create_counter(
                "agentproxy.api.requests",
                description="API requests by provider and model",
                unit="1",
            )
            self.api_errors = self.meter.create_counter(
                "agentproxy.api.errors",
                description="API errors by provider and type",
                unit="1",
            )
            self.api_cost = self.meter.create_counter(
                "agentproxy.api.cost",
                description="Estimated API cost in USD",
                unit="usd",
            )

            # Context window tracking
            self.context_window_usage = self.meter.create_histogram(
                "agentproxy.context_window.usage_percent",
                description="Context window usage as percentage of model limit",
                unit="%",
            )

            # Multi-worker coordination tracking
            self.milestones_dispatched = self.meter.create_counter(
                "agentproxy.milestones.dispatched",
                description="Number of milestones dispatched to workers",
                unit="1",
            )
            self.milestones_completed = self.meter.create_counter(
                "agentproxy.milestones.completed",
                description="Number of milestones completed by workers",
                unit="1",
            )
            self.milestone_duration = self.meter.create_histogram(
                "agentproxy.milestone.duration",
                description="Milestone execution duration",
                unit="s",
            )

            # Code change tracking
            self.code_lines_added = self.meter.create_counter(
                "agentproxy.code.lines_added",
                description="Lines of code added",
                unit="lines",
            )
            self.code_lines_removed = self.meter.create_counter(
                "agentproxy.code.lines_removed",
                description="Lines of code removed",
                unit="lines",
            )
            self.code_files_modified = self.meter.create_counter(
                "agentproxy.code.files_modified",
                description="Files modified count",
                unit="1",
            )

        def instrument_fastapi(self, app):
            """Auto-instrument FastAPI if telemetry enabled"""
            if self.enabled and app is not None:
                FastAPIInstrumentor.instrument_app(app)


def calculate_cost(api: str, model: str, prompt_tokens: int, completion_tokens: int,
                   cache_write: int = 0, cache_read: int = 0) -> float:
    """Calculate estimated API cost."""
    pricing = API_PRICING.get(api, {}).get(model)
    if not pricing:
        return 0.0

    cost = (prompt_tokens / 1_000_000) * pricing.get("prompt", 0)
    cost += (completion_tokens / 1_000_000) * pricing.get("completion", 0)
    cost += (cache_write / 1_000_000) * pricing.get("cache_write", 0)
    cost += (cache_read / 1_000_000) * pricing.get("cache_read", 0)

    return cost


# Global singleton
_telemetry: Optional[AgentProxyTelemetry] = None


def get_telemetry():
    """Get or create global telemetry instance"""
    global _telemetry

    if not OTEL_AVAILABLE:
        # OTEL not installed, return no-op
        if _telemetry is None:
            _telemetry = NoOpTelemetry()
            # Only print if telemetry was requested
            if os.getenv("AGENTPROXY_ENABLE_TELEMETRY", "0") == "1":
                print("\033[2m│ 📊 OTEL     │\033[0m \033[91mOTEL packages not installed\033[0m")
                print("\033[2m│ 📊 OTEL     │\033[0m Install: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc")
        return _telemetry

    if _telemetry is None:
        _telemetry = AgentProxyTelemetry()
    return _telemetry


def reset_telemetry():
    """Reset global telemetry instance (for testing)"""
    global _telemetry
    _telemetry = None


def flush_telemetry():
    """Force flush all pending spans and metrics."""
    if not OTEL_AVAILABLE:
        return

    try:
        # Flush traces
        from opentelemetry import trace
        tracer_provider = trace.get_tracer_provider()
        if hasattr(tracer_provider, 'force_flush'):
            tracer_provider.force_flush()

        # Flush metrics
        from opentelemetry import metrics
        meter_provider = metrics.get_meter_provider()
        if hasattr(meter_provider, 'force_flush'):
            meter_provider.force_flush()

        telemetry = get_telemetry()
        telemetry.log("Flushed telemetry data to collector")
    except Exception as e:
        # Silent fail - telemetry shouldn't break the app
        pass
