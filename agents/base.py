"""
LLM factory with per-agent routing, automatic fallback chains,
and robust error handling (429 rate limits, timeouts, JSON failures).

Routing table:
    ┌──────────────┬─────────────────────────────────────────────┬──────────────────────────────────────────┐
    │ Agent        │ Primary                                     │ Fallback                                 │
    ├──────────────┼─────────────────────────────────────────────┼──────────────────────────────────────────┤
    │ Metrics      │ Llama-3.1-8B (HF Inference / Scaleway)      │ Llama-3.1-8B (Groq)                      │
    │ Risk         │ Qwen-2.5-72B (HF Inference / Novita)        │ Llama-3.3-70B-Instruct:free (OpenRouter)  │
    │ News         │ gemini-3-flash-preview (Google AI)           │ gemini-2.5-flash (Google AI)              │
    │ Supervisor   │ gpt-oss-120b (OpenRouter)                   │ Qwen-2.5-72B (HF Inference)              │
    │ Synthesis    │ gpt-oss-120b (Cerebras)                     │ Qwen3-32B (Groq)                         │
    └──────────────┴─────────────────────────────────────────────┴──────────────────────────────────────────┘

Fallback triggers:
    • Provider rate limit (HTTP 429) → switch to fallback immediately
    • Request timeout (> 10 seconds) → switch to faster fallback model
    • Malformed JSON / Pydantic validation failure → retry on fallback
    • Full provider outage (connection error) → entire chain falls to next
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, TypeVar

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from callbacks import get_langfuse_handler
from config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# ── Provider base URLs ───────────────────────────────────────────────────────
HF_INFERENCE_URL = "https://router.huggingface.co/v1"
OPENROUTER_URL = "https://openrouter.ai/api/v1"
GROQ_URL = "https://api.groq.com/openai/v1"
CEREBRAS_URL = "https://api.cerebras.ai/v1"
GOOGLE_GENAI_URL = "https://generativelanguage.googleapis.com/v1beta/openai"

# ── Timeouts ─────────────────────────────────────────────────────────────────
REQUEST_TIMEOUT = 30   # seconds — triggers fallback if exceeded
FALLBACK_TIMEOUT = 45  # slightly higher for fallback models


# ── Model definitions ────────────────────────────────────────────────────────


@dataclass
class ModelConfig:
    """Configuration for a single LLM model."""

    model_id: str
    provider: str        # "hf_inference", "openrouter", "groq", "cerebras", "google"
    max_tokens: int = 4096
    temperature: float = 0.0
    timeout: int = REQUEST_TIMEOUT


# Agent → (primary, fallback) model chains
AGENT_MODELS: dict[str, list[ModelConfig]] = {
    "metrics": [
        ModelConfig("meta-llama/Llama-3.1-8B-Instruct:scaleway", "hf_inference"),
        ModelConfig("llama-3.1-8b-instant", "groq"),
    ],
    "risk": [
        ModelConfig("Qwen/Qwen2.5-72B-Instruct:novita", "hf_inference"),
        ModelConfig("meta-llama/llama-3.3-70b-instruct:free", "openrouter"),
    ],
    "news": [
        ModelConfig("gemini-3-flash-preview", "google"),
        ModelConfig("gemini-2.5-flash", "google"),
    ],
    "supervisor": [
        ModelConfig("openai/gpt-oss-120b", "openrouter"),
        ModelConfig("Qwen/Qwen2.5-72B-Instruct:novita", "hf_inference"),
    ],
    "synthesis": [
        ModelConfig("gpt-oss-120b", "cerebras"),
        ModelConfig("qwen/qwen3-32b", "groq"),
    ],
}


def _get_api_key(provider: str) -> str:
    """Get the API key for a given provider."""
    key_map = {
        "hf_inference": settings.hf_token,
        "openrouter": settings.openrouter_api_key,
        "groq": settings.groq_api_key,
        "cerebras": settings.cerebras_api_key,
        "google": settings.google_api_key,
    }
    return key_map.get(provider, "")


def _get_base_url(provider: str) -> str:
    """Get the base URL for a given provider."""
    url_map = {
        "hf_inference": HF_INFERENCE_URL,
        "openrouter": OPENROUTER_URL,
        "groq": GROQ_URL,
        "cerebras": CEREBRAS_URL,
        "google": GOOGLE_GENAI_URL,
    }
    return url_map.get(provider, "")


def _is_rate_limit_error(exc: Exception) -> bool:
    """Check if an exception is a 429 rate limit error."""
    exc_str = str(exc).lower()
    return "429" in exc_str or "rate limit" in exc_str or "rate_limit" in exc_str


def _is_timeout_error(exc: Exception) -> bool:
    """Check if an exception is a timeout error."""
    exc_str = str(exc).lower()
    return "timeout" in exc_str or "timed out" in exc_str


def _is_json_error(exc: Exception) -> bool:
    """Check if an exception is a JSON parsing / Pydantic validation error."""
    return isinstance(exc, (ValidationError, ValueError)) or "json" in str(exc).lower()


def _classify_error(exc: Exception) -> str:
    """Classify an error for logging purposes."""
    if _is_rate_limit_error(exc):
        return "RATE_LIMIT_429"
    if _is_timeout_error(exc):
        return "TIMEOUT"
    if _is_json_error(exc):
        return "JSON_VALIDATION"
    return "PROVIDER_ERROR"


def _create_llm(config: ModelConfig) -> ChatOpenAI:
    """Create a ChatOpenAI instance from a ModelConfig."""
    api_key = _get_api_key(config.provider)
    base_url = _get_base_url(config.provider)

    if not api_key:
        raise ValueError(f"No API key for provider '{config.provider}'")

    return ChatOpenAI(
        model=config.model_id,
        openai_api_key=api_key,
        openai_api_base=base_url,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        request_timeout=config.timeout,
    )


def get_llm(agent_name: str) -> ChatOpenAI:
    """Get the best available LLM for the given agent.

    Tries the primary model first. If the API key is missing,
    falls back immediately (no network call).

    Args:
        agent_name: One of 'metrics', 'risk', 'news', 'supervisor', 'synthesis'.

    Returns:
        A ChatOpenAI instance configured for the correct provider.

    Raises:
        RuntimeError: If all models in the chain fail.
    """
    models = AGENT_MODELS.get(agent_name)
    if not models:
        raise ValueError(f"Unknown agent: '{agent_name}'. Valid: {list(AGENT_MODELS)}")

    for config in models:
        try:
            llm = _create_llm(config)
            logger.info(
                "Agent '%s' → %s via %s",
                agent_name,
                config.model_id,
                config.provider,
            )
            return llm
        except ValueError as exc:
            logger.warning(
                "Agent '%s' skipping %s (%s): %s",
                agent_name,
                config.model_id,
                config.provider,
                exc,
            )
            continue

    raise RuntimeError(
        f"All LLM models failed for agent '{agent_name}'. "
        f"Check your API keys in .env."
    )


def invoke_with_fallback(
    agent_name: str,
    prompt_chain: Any,
    input_data: dict,
    output_schema: type[T],
    config: dict | None = None,
) -> T:
    """Invoke an LLM chain with automatic fallback on failure.

    This is the robust invocation function that handles:
    - 429 rate limits → immediate fallback
    - Timeouts → fallback to faster model
    - JSON/Pydantic errors → retry on fallback with better compliance
    - Connection errors → fall through entire chain

    Args:
        agent_name: Agent name for model selection.
        prompt_chain: A LangChain prompt template (PROMPT).
        input_data: Dict of prompt variables.
        output_schema: Pydantic model class for structured output.
        config: Optional LangChain config (Langfuse callbacks, etc.).

    Returns:
        An instance of output_schema.

    Raises:
        RuntimeError: If all models in the fallback chain fail.
    """
    models = AGENT_MODELS.get(agent_name)
    if not models:
        raise ValueError(f"Unknown agent: '{agent_name}'")

    last_error: Exception | None = None

    for i, model_config in enumerate(models):
        is_fallback = i > 0
        label = "fallback" if is_fallback else "primary"

        try:
            api_key = _get_api_key(model_config.provider)
            if not api_key:
                logger.warning(
                    "[%s] %s: No API key for %s — skipping",
                    agent_name, label, model_config.provider,
                )
                continue

            llm = _create_llm(model_config)
            structured_llm = llm.with_structured_output(output_schema)
            chain = prompt_chain | structured_llm

            logger.info(
                "[%s] %s: Trying %s via %s",
                agent_name, label, model_config.model_id, model_config.provider,
            )

            start = time.monotonic()
            result = chain.invoke(input_data, config=config)
            elapsed = time.monotonic() - start

            logger.info(
                "[%s] %s: ✅ Success (%s, %.1fs)",
                agent_name, label, model_config.model_id, elapsed,
            )
            return result

        except Exception as exc:
            error_type = _classify_error(exc)
            elapsed = time.monotonic() - start if 'start' in dir() else 0

            logger.warning(
                "[%s] %s: ❌ %s from %s (%.1fs): %s",
                agent_name, label, error_type,
                model_config.model_id, elapsed, str(exc)[:200],
            )
            last_error = exc
            continue

    raise RuntimeError(
        f"All models failed for agent '{agent_name}'. "
        f"Last error: {last_error}"
    )


def create_langfuse_config(
    session_id: str = "",
    trace_name: str = "",
    **kwargs,
) -> dict:
    """Create a LangChain config dict with Langfuse tracing attached.

    Args:
        session_id: Groups related traces into a session.
        trace_name: Custom name shown in Langfuse UI.

    Returns:
        Dict ready to pass as config= to .invoke() / .ainvoke().
    """
    handler = get_langfuse_handler(
        session_id=session_id,
        trace_name=trace_name,
        **kwargs,
    )
    return {"callbacks": [handler]}
