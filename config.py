"""
Central configuration — loads .env and validates all required API keys.

Usage:
    from config import settings
    print(settings.openrouter_api_key)
"""

import os
import sys
from pathlib import Path

# Ensure project root is in the python path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# ── Load .env from project root ─────────────────────────────────────────────
load_dotenv(_PROJECT_ROOT / ".env")


class Settings(BaseModel):
    """Validated application settings sourced from environment variables."""

    # ── Langfuse Observability ───────────────────────────────────────────────
    langfuse_secret_key: str = Field(default="", description="Langfuse secret key")
    langfuse_public_key: str = Field(default="", description="Langfuse public key")
    langfuse_base_url: str = Field(
        default="https://us.cloud.langfuse.com",
        description="Langfuse host URL",
    )

    # ── LLM Providers ─────────────────────────────────────────────────────
    openrouter_api_key: str = Field(
        default="", description="OpenRouter API key"
    )
    google_api_key: str = Field(default="", description="Google AI API key (primary)")
    google_api_keys: list[str] = Field(
        default_factory=list,
        description="All Google API keys for rotation on rate limits",
    )
    groq_api_key: str = Field(default="", description="Groq API key (Llama fallback)")
    cerebras_api_key: str = Field(default="", description="Cerebras API key (gpt-oss-120b)")
    nvidia_api_key: str = Field(default="", description="NVIDIA API key (Nemotron fallback)")

    # ── OpenAI-compatible fallback ───────────────────────────────────────
    openai_api_key: str = Field(default="", description="OpenAI-compat API key")
    openai_base_url: str = Field(default="", description="OpenAI-compat base URL")

    # ── Embeddings ───────────────────────────────────────────────────────────
    hf_token: str = Field(default="", description="HuggingFace token for embeddings")
    voyage_api_key: str = Field(default="", description="Voyage AI API key for embeddings")
    nomic_api_key: str = Field(default="", description="Nomic Atlas API key for embeddings")

    # ── Ingestion ────────────────────────────────────────────────────────────
    jina_api_key: str = Field(default="", description="Jina Reader API key")

    # ── Vector Store ─────────────────────────────────────────────────────────
    qdrant_url: str = Field(
        default="http://localhost:6333", description="Qdrant server URL"
    )
    qdrant_api_key: str = Field(default="", description="Qdrant API key (if secured)")

    # ── Session Store & Database ─────────────────────────────────────────────
    redis_url: str = Field(
        default="redis://localhost:6379", description="Redis URL for session persistence"
    )
    supabase_uri: str = Field(
        default="", description="Supabase Postgres connection URI"
    )

    # ── Web Search ───────────────────────────────────────────────────────────
    tavily_api_key: str = Field(default="", description="Tavily API key for News Agent")

    @classmethod
    def from_env(cls) -> "Settings":
        """Construct Settings by reading current environment variables."""
        # Collect all GOOGLE_API_KEY variants
        google_keys = []
        primary_google = os.getenv("GOOGLE_API_KEY", "")
        if primary_google:
            google_keys.append(primary_google)
        # Scan for GOOGLE_API_KEY_2, _3, ..., _10
        for i in range(2, 11):
            key = os.getenv(f"GOOGLE_API_KEY_{i}", "")
            if key:
                google_keys.append(key)
        # Also check the typo variant
        typo_key = os.getenv("GOOGLW_API_KEY_3", "")
        if typo_key and typo_key not in google_keys:
            google_keys.append(typo_key)

        return cls(
            langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            langfuse_base_url=os.getenv(
                "LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com"
            ),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
            google_api_key=primary_google,
            google_api_keys=google_keys,
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
            cerebras_api_key=os.getenv("CEREBRAS_API_KEY", ""),
            nvidia_api_key=os.getenv("NVIDIA_API_KEY", ""),
            openai_api_key=os.getenv("API_KEY", ""),
            openai_base_url=os.getenv("BASE_URL", ""),
            hf_token=os.getenv("HF_TOKEN", ""),
            voyage_api_key=os.getenv("VOYAGE_API_KEY", ""),
            nomic_api_key=os.getenv("NOMIC_API_KEY", ""),
            jina_api_key=os.getenv("JINA_API_KEY", ""),
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY", ""),
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
            supabase_uri=f"postgresql+asyncpg://postgres.cakvdlozssbxxpfvsrsj:{os.getenv('DB_PASS', '')}@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres",
            tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
        )

    def check_keys(self) -> tuple[dict[str, bool], dict[str, bool]]:
        """Return (required, optional) dicts mapping key name → is_present."""
        required = {
            "LANGFUSE_SECRET_KEY": bool(self.langfuse_secret_key),
            "LANGFUSE_PUBLIC_KEY": bool(self.langfuse_public_key),
            "OPENROUTER_API_KEY": bool(self.openrouter_api_key),
            "HF_TOKEN": bool(self.hf_token),
            "JINA_API_KEY": bool(self.jina_api_key),
        }
        optional = {
            "LANGFUSE_BASE_URL": bool(self.langfuse_base_url),
            "GOOGLE_API_KEY": bool(self.google_api_key),
            "API_KEY (OpenAI-compat)": bool(self.openai_api_key),
            "TAVILY_API_KEY": bool(self.tavily_api_key),
            "QDRANT_URL": bool(self.qdrant_url),
        }
        return required, optional


# ── Singleton — import this everywhere ───────────────────────────────────────
settings = Settings.from_env()

# ── Logging Setup ─────────────────────────────────────────────────────────────
import logging
import logging.handlers
import os
import sys
from datetime import datetime as _dt


def _setup_logging(process_name: str = "app") -> None:
    """Configure production-grade logging for a named process.

    Creates:
        logs/<process_name>_YYYY-MM-DD_HH-MM-SS.log  — timestamped session log
        logs/<process_name>_latest.log                — symlink → latest session

    Third-party loggers are silenced to WARNING so our own INFO logs
    stay readable without being buried in httpx/watchfiles/sqlalchemy noise.
    """
    log_dir = _PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    timestamp = _dt.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"{process_name}_{timestamp}.log"
    latest_link = log_dir / f"{process_name}_latest.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture everything; handlers filter

    # Avoid duplicate handlers if imported multiple times
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)-8s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler: INFO+ from our code, WARNING+ from libraries ─────────
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    root_logger.addHandler(console)

    # ── File handler: DEBUG+ everything goes to file ───────────────────────────
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=25 * 1024 * 1024,  # 25 MB per file
        backupCount=10,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root_logger.addHandler(file_handler)

    # ── Symlink: logs/<process>_latest.log → current session file ─────────────
    try:
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(log_file.name)
    except Exception:
        pass  # Non-critical — Windows or permission issue

    # ── Silence noisy third-party loggers (keep WARNING+ only) ────────────────
    _NOISY_LOGGERS = [
        "httpx", "httpcore", "watchfiles", "watchgod",
        "sqlalchemy", "sqlalchemy.engine", "sqlalchemy.pool",
        "asyncio", "urllib3", "filelock", "PIL",
        "langfuse", "openai", "anthropic",
        "chainlit.server", "chainlit.socket", "uvicorn.access",
    ]
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    # Log startup banner to file
    startup_logger = logging.getLogger("startup")
    startup_logger.info("=" * 70)
    startup_logger.info("Process : %s", process_name)
    startup_logger.info("Log file: %s", log_file)
    startup_logger.info("Started : %s", _dt.now().isoformat())
    startup_logger.info("Python  : %s", sys.version.split()[0])
    startup_logger.info("=" * 70)


def _detect_process_name() -> str:
    """Detect whether we are running under chainlit, uvicorn, or plain python."""
    cmd = " ".join(sys.argv).lower()
    if "chainlit" in cmd:
        return "chainlit"
    if "uvicorn" in cmd or "api" in cmd:
        return "api"
    if "pytest" in cmd or "test" in cmd:
        return "test"
    return "app"


_setup_logging(_detect_process_name())

