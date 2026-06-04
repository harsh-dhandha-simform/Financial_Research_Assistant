"""
LLM factory with per-agent routing and automatic fallback chains.

Centralises all LLM model creation so every agent uses the correct
provider, has Langfuse tracing, and fails over gracefully.

Routing table:
    ┌──────────────┬─────────────────────────────────────────────┬──────────────────────────────────────────┐
    │ Agent        │ Primary                                     │ Fallback                                 │
    ├──────────────┼─────────────────────────────────────────────┼──────────────────────────────────────────┤
    │ Metrics      │ Llama-3.1-8B (HF Inference / Scaleway)      │ Llama-3.1-8B (Groq)                      │
    │ Risk         │ Qwen-2.5-72B (HF Inference / Novita)        │ Llama-3.3-70B-Instruct:free (OpenRouter)  │
    │ News         │ gemini-3-flash-preview (Google AI)           │ gemini-2.5-flash (Google AI)              │
    │ Supervisor   │ gpt-oss-120b (OpenRouter)                   │ Qwen-2.5-72B (HF Inference)              │
    │ Synthesis    │ gpt-oss-120b (Cerebras)                     │ Qwen3-32B (OpenRouter)                   │
    └──────────────┴─────────────────────────────────────────────┴──────────────────────────────────────────┘
"""

import logging
from dataclasses import dataclass

from langchain_openai import ChatOpenAI

from callbacks import get_langfuse_handler
from config import settings

logger = logging.getLogger(__name__)


# ── Provider base URLs ───────────────────────────────────────────────────────
HF_INFERENCE_URL = "https://router.huggingface.co/v1"
OPENROUTER_URL = "https://openrouter.ai/api/v1"
GROQ_URL = "https://api.groq.com/openai/v1"
CEREBRAS_URL = "https://api.cerebras.ai/v1"
GOOGLE_GENAI_URL = "https://generativelanguage.googleapis.com/v1beta/openai"


# ── Model definitions ────────────────────────────────────────────────────────


@dataclass
class ModelConfig:
    """Configuration for a single LLM model."""

    model_id: str
    provider: str        # "hf_inference", "openrouter", "groq", "cerebras", "google"
    max_tokens: int = 4096
    temperature: float = 0.0


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
        ModelConfig("qwen/qwen3-32b", "openrouter"),
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
    )


def get_llm(agent_name: str) -> ChatOpenAI:
    """Get the best available LLM for the given agent, with automatic fallback.

    Tries the primary model first. If it fails (missing API key, etc.),
    falls back to the next model in the chain.

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
