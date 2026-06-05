"""
extract_financial_table — extracts structured financial metrics from text.

Parses HTML tables and raw text financial tables into structured dicts.
Triggered when retrieved chunks contain $ or % indicators.

Used by: Metrics Agent.
"""

import logging
import re
from typing import Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# ── Financial metric patterns ────────────────────────────────────────────────
# Matches patterns like "$394.3 billion", "25.3%", "$6.08", etc.
CURRENCY_PATTERN = re.compile(
    r"\$\s*([\d,]+(?:\.\d+)?)\s*(billion|million|thousand|B|M|K)?",
    re.IGNORECASE,
)
PERCENT_PATTERN = re.compile(
    r"([\d]+(?:\.\d+)?)\s*%",
)

# Named metric patterns (case-insensitive)
METRIC_PATTERNS = {
    "revenue": re.compile(
        r"(?:total\s+)?(?:net\s+)?(?:revenue|sales)\s*(?:was|of|:)?\s*\$?\s*([\d,]+(?:\.\d+)?)\s*(billion|million|B|M)?",
        re.IGNORECASE,
    ),
    "net_income": re.compile(
        r"net\s+income\s*(?:was|of|:)?\s*\$?\s*([\d,]+(?:\.\d+)?)\s*(billion|million|B|M)?",
        re.IGNORECASE,
    ),
    "gross_margin": re.compile(
        r"gross\s+(?:profit\s+)?margin\s*(?:was|of|:)?\s*([\d]+(?:\.\d+)?)\s*%",
        re.IGNORECASE,
    ),
    "operating_margin": re.compile(
        r"operating\s+margin\s*(?:was|of|:)?\s*([\d]+(?:\.\d+)?)\s*%",
        re.IGNORECASE,
    ),
    "ebitda": re.compile(
        r"EBITDA\s*(?:was|of|:)?\s*\$?\s*([\d,]+(?:\.\d+)?)\s*(billion|million|B|M)?",
        re.IGNORECASE,
    ),
    "eps": re.compile(
        r"(?:diluted\s+)?(?:earnings?\s+per\s+share|EPS)\s*(?:was|of|:)?\s*\$?\s*([\d]+(?:\.\d+)?)",
        re.IGNORECASE,
    ),
    "debt_equity": re.compile(
        r"debt[\s-]+(?:to[\s-]+)?equity\s*(?:ratio)?\s*(?:was|of|:)?\s*([\d]+(?:\.\d+)?)",
        re.IGNORECASE,
    ),
    "pe_ratio": re.compile(
        r"(?:P/?E|price[\s-]+(?:to[\s-]+)?earnings)\s*(?:ratio)?\s*(?:was|of|:)?\s*([\d]+(?:\.\d+)?)",
        re.IGNORECASE,
    ),
    "free_cash_flow": re.compile(
        r"free\s+cash\s+flow\s*(?:was|of|:)?\s*\$?\s*([\d,]+(?:\.\d+)?)\s*(billion|million|B|M)?",
        re.IGNORECASE,
    ),
    "roe": re.compile(
        r"(?:return\s+on\s+equity|ROE)\s*(?:was|of|:)?\s*([\d]+(?:\.\d+)?)\s*%",
        re.IGNORECASE,
    ),
}


def _normalize_value(value: str, unit: str | None = None) -> str:
    """Normalize a financial value with its unit."""
    cleaned = value.replace(",", "")
    if unit:
        unit_lower = unit.lower()
        if unit_lower in ("billion", "b"):
            return f"${cleaned}B"
        elif unit_lower in ("million", "m"):
            return f"${cleaned}M"
        elif unit_lower in ("thousand", "k"):
            return f"${cleaned}K"
    return f"${cleaned}" if not cleaned.startswith("$") else cleaned


@tool
def extract_financial_table(text: str) -> dict[str, str | None]:
    """Extract structured financial metrics from text containing financial data.

    Parses both HTML tables and raw text to extract key financial metrics.
    Triggered when retrieved chunks contain $ or % indicators.

    Args:
        text: Raw retrieved chunk content potentially containing financial tables.

    Returns:
        Dict with metric names as keys and extracted values (or None if not found).
        Keys: revenue, net_income, gross_margin, ebitda, eps, pe_ratio,
              debt_equity, free_cash_flow, operating_margin, roe.
    """
    if not text or not text.strip():
        return {name: None for name in METRIC_PATTERNS}

    results: dict[str, str | None] = {}

    for name, pattern in METRIC_PATTERNS.items():
        match = pattern.search(text)
        if match:
            groups = match.groups()
            value = groups[0]
            unit = groups[1] if len(groups) > 1 else None

            if "%" in name or name in ("gross_margin", "operating_margin", "roe"):
                results[name] = f"{value}%"
            elif name in ("eps", "pe_ratio", "debt_equity"):
                results[name] = value
            else:
                results[name] = _normalize_value(value, unit)
        else:
            results[name] = None

    found = {k: v for k, v in results.items() if v is not None}
    logger.info(
        "extract_financial_table: found %d/%d metrics: %s",
        len(found),
        len(METRIC_PATTERNS),
        list(found.keys()),
    )
    return results
