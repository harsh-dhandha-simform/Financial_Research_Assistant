"""
Image describer — lazy vision description via Groq llama-4-scout.

Called ONLY at retrieval time when a chunk containing an image is surfaced.
NOT called during ingestion — ingestion only extracts + stores the raw image.

Descriptions are cached in a dict so the same image is never described twice.

Usage:
    from ingestion.image_describer import describe_image, classify_image_type

    description = describe_image("/path/to/image.png", context="Revenue chart")
    image_type = classify_image_type(description)
"""

import base64
import logging
import os
from typing import Optional
from langfuse.decorators import observe

logger = logging.getLogger(__name__)

# ── Description cache: image_path → description ─────────────────────────────
_description_cache: dict[str, str] = {}

# ── Image type cache: image_path → type ─────────────────────────────────────
_type_cache: dict[str, str] = {}


def _get_groq_client():
    """Get or create the Groq client."""
    from groq import Groq
    from config import settings

    if not settings.groq_api_key:
        raise ValueError("GROQ_API_KEY not set — cannot use vision model")

    return Groq(api_key=settings.groq_api_key)


def _image_to_base64_url(image_path: str) -> str:
    """Convert a local image file to a base64 data URL."""
    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    mime_map = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
        "gif": "image/gif",
    }
    mime = mime_map.get(ext, "image/png")

    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime};base64,{data}"


@observe(as_type="generation")
def describe_image(
    image_path_or_url: str,
    context: str = "",
    force: bool = False,
) -> str:
    """Describe a financial document image using Groq vision.

    Uses meta-llama/llama-4-scout-17b-16e-instruct via Groq.
    Results are cached — repeated calls for the same image return
    the cached description without another API call.

    Args:
        image_path_or_url: Local file path or HTTP URL of the image.
        context: Optional surrounding text context for better description.
        force: If True, bypass cache and re-describe.

    Returns:
        Description string of the image contents.
    """
    # Check cache first
    if not force and image_path_or_url in _description_cache:
        logger.debug("Image description cache hit: %s", image_path_or_url)
        return _description_cache[image_path_or_url]

    # Build the image content
    if image_path_or_url.startswith(("http://", "https://", "data:")):
        image_url = image_path_or_url
    else:
        # Local file → base64
        if not os.path.exists(image_path_or_url):
            logger.warning("Image file not found: %s", image_path_or_url)
            return "Image file not found."
        image_url = _image_to_base64_url(image_path_or_url)

    prompt = (
        "Describe this financial document image in detail. "
        "If it contains a table, extract all values and column headers. "
        "If it contains a chart or graph, describe its type, axes, and key data points. "
        "If it contains text, transcribe it accurately."
    )
    if context:
        prompt += f"\n\nContext from surrounding document: {context}"

    try:
        client = _get_groq_client()

        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                    ],
                }
            ],
            max_tokens=1024,
            temperature=0.1,
        )

        description = response.choices[0].message.content or ""
        description = description.strip()

        # Cache the result
        _description_cache[image_path_or_url] = description

        logger.info(
            "Described image: %s (%d chars)",
            os.path.basename(image_path_or_url) if not image_path_or_url.startswith("http") else image_path_or_url[:60],
            len(description),
        )
        return description

    except Exception as exc:
        logger.error("Vision description failed for %s: %s", image_path_or_url, exc)
        return f"[Image description unavailable: {exc}]"


def classify_image_type(description: str) -> str:
    """Classify an image type based on its description.

    Returns one of: "table", "chart", "graph", "diagram", "other"
    """
    description_lower = description.lower()

    if any(kw in description_lower for kw in ["table", "column", "row", "header", "cell"]):
        return "table"
    if any(kw in description_lower for kw in ["bar chart", "pie chart", "histogram", "chart"]):
        return "chart"
    if any(kw in description_lower for kw in ["graph", "line graph", "scatter", "trend"]):
        return "graph"
    if any(kw in description_lower for kw in ["diagram", "flowchart", "org chart", "architecture"]):
        return "diagram"

    return "other"


def get_cached_description(image_path: str) -> Optional[str]:
    """Get a cached description if available. Returns None if not cached."""
    return _description_cache.get(image_path)


def clear_cache():
    """Clear all cached descriptions."""
    _description_cache.clear()
    _type_cache.clear()
