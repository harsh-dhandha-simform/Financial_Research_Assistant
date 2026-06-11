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
    message_content = f"📎 **{len(citations)} source(s)**\n"
    for i, cit in enumerate(citations):
        name = f"📄 Source {i+1}: {cit.source_document} p.{cit.page_number}"
        message_content += f"- [[{name}]]\n"
        
        # Build content for the citation side panel
        content = (
            f"**Document:** {cit.source_document}\n"
            f"**Page:** {cit.page_number} | **Section:** {cit.section}\n"
            f"**Relevance:** {cit.confidence:.0%}\n\n"
            f"> {cit.chunk_preview}...\n"
        )
        
        # Add image element and reference it in details if available
        if cit.has_image and cit.image_url:
            content += (
                f"\n<details>\n"
                f"<summary>view source images</summary>\n\n"
                f"[[Source Image {i+1}]]\n"
                f"</details>\n"
            )
            if cit.image_url.startswith("http"):
                img_el = cl.Image(
                    name=f"Source Image {i+1}",
                    url=cit.image_url,
                    display="inline"
                )
            else:
                img_el = cl.Image(
                    name=f"Source Image {i+1}",
                    path=cit.image_url,
                    display="inline"
                )
            elements.append(img_el)
            
        elements.append(cl.Text(
            name=name,
            content=content,
            display="side"
        ))

    await cl.Message(
        content=message_content,
        elements=elements
    ).send()
