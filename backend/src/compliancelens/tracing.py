"""OpenTelemetry → Arize Phoenix (local OTLP/HTTP collector)."""

from __future__ import annotations

import logging

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from compliancelens.config import Settings

log = logging.getLogger(__name__)
_configured = False


def setup_tracing(settings: Settings) -> None:
    """Export spans to Phoenix and auto-instrument the OpenAI/Anthropic SDKs.

    Idempotent; a no-op when ``TRACING_ENABLED=false``. Phoenix being down never
    breaks requests — the batch exporter just logs and drops.
    """
    global _configured
    if _configured or not settings.tracing_enabled:
        return
    resource = Resource.create(
        {
            "service.name": "compliancelens-api",
            "openinference.project.name": settings.phoenix_project_name,
        }
    )
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.phoenix_collector_endpoint))
    )
    trace.set_tracer_provider(provider)

    try:
        from openinference.instrumentation.anthropic import AnthropicInstrumentor
        from openinference.instrumentation.openai import OpenAIInstrumentor

        OpenAIInstrumentor().instrument(tracer_provider=provider)
        AnthropicInstrumentor().instrument(tracer_provider=provider)
    except ImportError:  # pragma: no cover
        log.warning("openinference instrumentors not installed; only manual spans exported")
    _configured = True
