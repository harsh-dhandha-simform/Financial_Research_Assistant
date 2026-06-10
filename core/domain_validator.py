"""
Domain validator — URL validation for financial document ingestion.

Two-stage validation:
  Stage 1 (pre-fetch): Check domain → reject if obviously non-financial.
  Stage 2 (post-fetch): Pass first 2000 chars to LLM for content verification.

Usage:
    from core.domain_validator import is_edgar_url, is_known_financial_domain, validate_url_domain, validate_document_content
"""

import logging
from urllib.parse import urlparse
from langfuse.decorators import observe

logger = logging.getLogger(__name__)


# ── Known patterns and domains ───────────────────────────────────────────────

ALLOWED_EDGAR_PATTERNS = [
    "sec.gov/Archives/edgar/",
    "sec.gov/cgi-bin/browse-edgar",
    "efts.sec.gov",
]

KNOWN_FINANCIAL_DOMAINS = [
    "sec.gov", "bloomberg.com", "reuters.com", "wsj.com",
    "ft.com", "morningstar.com", "marketwatch.com",
    "finance.yahoo.com", "seekingalpha.com", "investing.com",
    "macrotrends.net", "annualreports.com", "last10k.com",
    "ir.",       # IR subdomains (e.g. ir.apple.com)
    "investor.", # Investor subdomains (e.g. investor.apple.com)
]

# Domains that are NEVER financial — immediate reject
BLOCKED_DOMAINS = [
    "youtube.com", "tiktok.com", "instagram.com", "facebook.com",
    "twitter.com", "x.com", "reddit.com", "pinterest.com",
    "amazon.com", "ebay.com", "netflix.com", "spotify.com",
    "wikipedia.org", "stackoverflow.com", "github.com",
]


# ── Stage 1: Domain-level checks ────────────────────────────────────────────


def is_edgar_url(url: str) -> bool:
    """Check if a URL points to an SEC EDGAR filing."""
    return any(p in url for p in ALLOWED_EDGAR_PATTERNS)


def is_known_financial_domain(url: str) -> bool:
    """Check if a URL belongs to a known financial domain."""
    try:
        netloc = urlparse(url).netloc.lower()
    except Exception:
        return False
    return any(
        netloc.endswith(d) or netloc.startswith(d)
        for d in KNOWN_FINANCIAL_DOMAINS
    )


def is_blocked_domain(url: str) -> bool:
    """Check if a URL belongs to a known non-financial domain."""
    try:
        netloc = urlparse(url).netloc.lower()
    except Exception:
        return False
    return any(netloc.endswith(d) for d in BLOCKED_DOMAINS)


@observe()
def validate_url_domain(url: str) -> tuple[bool, str]:
    """Stage 1: Pre-fetch domain validation.

    Returns:
        (is_valid, reason) — if invalid, reason explains why.
    """
    if not url or not url.strip():
        return False, "URL is empty."

    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False, f"Invalid URL scheme: {parsed.scheme}. Use http or https."
        if not parsed.netloc:
            return False, "Invalid URL: no domain found."
    except Exception as exc:
        return False, f"URL parsing failed: {exc}"

    # EDGAR URLs are always valid
    if is_edgar_url(url):
        return True, "SEC EDGAR URL — accepted."

    # Blocked domains are always rejected
    if is_blocked_domain(url):
        return False, (
            "This URL does not appear to contain financial research content "
            "relevant to our system. Ingestion rejected."
        )

    # Known financial domains are accepted
    if is_known_financial_domain(url):
        return True, "Known financial domain — accepted."

    # Unknown domain — allow but flag for Stage 2 content validation
    return True, "Unknown domain — will validate content after fetch."


# ── Stage 2: Content-level validation (LLM) ─────────────────────────────────


@observe(as_type="generation")
async def validate_document_content(
    content_preview: str,
    url: str = "",
) -> tuple[bool, str]:
    """Stage 2: Post-fetch content validation using groq/llama-3.1-8b-instant.

    Passes the first 2000 chars to the LLM to determine if the document
    is a financial filing, annual report, prospectus, earnings release,
    or investment-relevant document.

    Args:
        content_preview: First 2000 chars of the fetched document.
        url: The source URL (for logging).

    Returns:
        (is_valid, reason)
    """
    from langchain_openai import ChatOpenAI
    from config import settings
    from callbacks import get_langfuse_handler

    if not settings.groq_api_key:
        logger.warning("GROQ_API_KEY not set — skipping content validation")
        return True, "Content validation skipped (no Groq key)."

    try:
        guardrail_llm = ChatOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=settings.groq_api_key,
            model="llama-3.1-8b-instant",
            temperature=0,
            max_tokens=150,
        )

        prompt = (
            "Is this document a financial filing, annual report, prospectus, "
            "earnings release, or investment-relevant document? "
            "Answer ONLY with JSON: {\"valid\": true/false, \"reason\": \"...\"}\n\n"
            f"Document preview (first 2000 chars):\n{content_preview[:2000]}"
        )

        handler = get_langfuse_handler(
            trace_name="url_validation",
            tags=["guardrail", "domain_validator"],
        )

        response = await guardrail_llm.ainvoke(
            prompt,
            config={"callbacks": [handler]},
        )

        import json
        try:
            result = json.loads(response.content)
            is_valid = result.get("valid", True)
            reason = result.get("reason", "No reason provided.")
        except (json.JSONDecodeError, AttributeError):
            # If LLM doesn't return valid JSON, check for YES/NO
            text = response.content.lower()
            is_valid = "yes" in text or "true" in text
            reason = response.content[:200]

        logger.info(
            "Content validation for %s: valid=%s, reason=%s",
            url[:60] if url else "unknown", is_valid, reason[:80],
        )
        return is_valid, reason

    except Exception as exc:
        logger.warning("Content validation failed: %s — allowing by default", exc)
        return True, f"Content validation error: {exc}"
