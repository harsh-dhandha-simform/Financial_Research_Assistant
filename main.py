"""
Module 1 — Project scaffolding verification.

Run with:  uv run python main.py
"""

from config import settings
from callbacks import get_langfuse_handler


def _mask(value: str) -> str:
    """Show first 8 chars of a secret, mask the rest."""
    if not value:
        return "(empty)"
    return value[:8] + "..." if len(value) > 8 else value


def verify_env() -> None:
    """Validate environment keys, Langfuse callback, and package structure."""

    print("=" * 60)
    print("  Financial Research Analyst — Environment Check")
    print("=" * 60)

    # ── Required keys ────────────────────────────────────────────────────────
    required, optional = settings.check_keys()
    all_ok = True

    print("\n── Required Keys ──")
    key_to_attr = {
        "LANGFUSE_SECRET_KEY": settings.langfuse_secret_key,
        "LANGFUSE_PUBLIC_KEY": settings.langfuse_public_key,
        "OPENROUTER_API_KEY": settings.openrouter_api_key,
        "HF_TOKEN": settings.hf_token,
        "JINA_API_KEY": settings.jina_api_key,
    }
    for name, present in required.items():
        value = key_to_attr[name]
        icon = "✅" if present else "❌ MISSING"
        if not present:
            all_ok = False
        print(f"  {icon}  {name}: {_mask(value)}")

    # ── Optional keys ────────────────────────────────────────────────────────
    print("\n── Optional Keys (needed later) ──")
    opt_attr = {
        "LANGFUSE_BASE_URL": settings.langfuse_base_url,
        "GOOGLE_API_KEY": settings.google_api_key,
        "API_KEY (OpenAI-compat)": settings.openai_api_key,
        "TAVILY_API_KEY": settings.tavily_api_key,
        "QDRANT_URL": settings.qdrant_url,
    }
    for name, present in optional.items():
        value = opt_attr[name]
        icon = "✅" if present else "⚠️  not set"
        print(f"  {icon}  {name}: {_mask(value)}")

    # ── Langfuse callback ────────────────────────────────────────────────────
    print("\n── Langfuse Callback ──")
    try:
        handler = get_langfuse_handler(trace_name="env-verification")
        print(f"  ✅  CallbackHandler created: {type(handler).__name__}")
    except Exception as exc:
        print(f"  ❌  Failed: {exc}")
        all_ok = False

    # ── Package structure ────────────────────────────────────────────────────
    print("\n── Package Structure ──")
    packages = [
        "schemas",
        "ingestion",
        "chunking",
        "retrieval",
        "agents",
        "graph",
        "output",
        "api",
        "chat",
    ]
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"  ✅  {pkg}/")
        except ImportError as exc:
            print(f"  ❌  {pkg}/ — {exc}")
            all_ok = False

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if all_ok:
        print("  ✅  All checks passed — ready for Module 2 (Pydantic schemas).")
    else:
        print("  ⚠️   Some checks failed. Fix the issues above, then re-run.")
    print("=" * 60)


if __name__ == "__main__":
    verify_env()
