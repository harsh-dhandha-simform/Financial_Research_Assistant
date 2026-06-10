"""
Source renderer — displays citation sources in Chainlit.

Renders Citation objects as collapsible side-panel elements
with image support.
"""

import chainlit as cl

from schemas.citation import Citation


async def render_sources(citations: list[Citation]):
    """Render a list of Citation objects as expandable source elements."""
    if not citations:
        return

    elements = []
    for i, cit in enumerate(citations):
        # Text source
        elements.append(cl.Text(
            name=f"📄 Source {i+1}: {cit.source_document} p.{cit.page_number}",
            content=(
                f"**Document:** {cit.source_document}\n"
                f"**Page:** {cit.page_number} | **Section:** {cit.section}\n"
                f"**Relevance:** {cit.confidence:.0%}\n\n"
                f"> {cit.chunk_preview}..."
            ),
            display="side"
        ))
        # Image source if available
        if cit.has_image and cit.image_url:
            if cit.image_url.startswith("http"):
                img_el = cl.Image(
                    name=f"🖼️ Visual ({cit.image_type})",
                    url=cit.image_url,
                    display="side"
                )
            else:
                img_el = cl.Image(
                    name=f"🖼️ Visual ({cit.image_type})",
                    path=cit.image_url,
                    display="side"
                )
            elements.append(img_el)

    await cl.Message(
        content=f"📎 **{len(citations)} source(s)** — click to expand",
        elements=elements
    ).send()
