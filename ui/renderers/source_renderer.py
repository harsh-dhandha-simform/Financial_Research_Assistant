"""
Source renderer — displays citation sources and document images in Chainlit.

Layout:
  ┌──────────────────────────────────────────────┐
  │  RAG answer (sent by _handle_chat)           │
  ├──────────────────────────────────────────────┤
  │  🖼️ Document Citations (this module)         │
  │  ┌────────┐  ┌────────┐  ┌────────┐          │
  │  │ crop 1 │  │ crop 2 │  │ crop 3 │  ← small │
  │  │ p.444  │  │ p.372  │  │ p.3    │    inline │
  │  └────────┘  └────────┘  └────────┘  ← click │
  │                                               │
  ├──────────────────────────────────────────────┤
  │  📎 N sources retrieved                      │
  │  · [[Source 1: doc p.X]]  · [Section]        │
  │  · [[Source 2: doc p.Y]]  · [Section]        │
  │  [📚 View Sources]                           │
  └──────────────────────────────────────────────┘

Images are shown as cl.Image(display="inline", size="small") — they render as
small inline thumbnails in the message that open full-size in a side panel on click.

Relevance scores: raw cosine similarity (dense_score), not normalised.
"""

import logging
import os

import chainlit as cl

from schemas.citation import Citation

logger = logging.getLogger(__name__)


# ── Section labels ────────────────────────────────────────────────────────────
_SECTION_LABELS: dict[str, str] = {
    "mda": "MD&A",
    "md&a": "MD&A",
    "risk_factors": "Risk Factors",
    "risk factors": "Risk Factors",
    "business_overview": "Business Overview",
    "business overview": "Business Overview",
    "notes": "Notes to Financials",
    "notes_to_financials": "Notes to Financials",
    "legal_proceedings": "Legal Proceedings",
    "legal proceedings": "Legal Proceedings",
    "financials": "Financial Statements",
    "financial_statements": "Financial Statements",
    "corporate_governance": "Corporate Governance",
    "forward_looking_statements": "Forward-Looking Statements",
    "general": "",
    "other": "",
}


def _fmt_section(section: str) -> str:
    if not section:
        return ""
    lower = section.lower().strip()
    return _SECTION_LABELS.get(lower, section.replace("_", " ").title())


def _relevance_chip(dense_score: float, rrf_score: float = 0.0) -> str:
    score = dense_score if dense_score and dense_score > 0.01 else rrf_score
    if score <= 0:
        return "—"
    pct = int(score * 100)
    if pct >= 80:
        color = "🟢"
    elif pct >= 60:
        color = "🟡"
    elif pct >= 40:
        color = "🟠"
    else:
        color = "🔴"
    return f"{color} {pct}%"


async def render_sources(citations: list[Citation]) -> None:
    """
    Render citation images inline below the RAG answer, then show the source list.

    Step 1 — Image preview message (if any images exist):
      Sends a single message containing all crop images as cl.Image(display="inline",
      size="small"). Each thumbnail is labelled with page number and doc name.
      Clicking any thumbnail opens it full-size in a side panel.

    Step 2 — Source list message:
      Compact list of [[Source N]] chips (opens metadata side panel) + a
      "📚 View Sources" action button to re-open the list on demand.
    """
    if not citations:
        return

    # ── Collect image elements ────────────────────────────────────────────────
    image_elements: list[cl.Image] = []
    image_lines: list[str] = []          # labels under the images

    for i, cit in enumerate(citations):
        doc_label = os.path.basename(cit.source_document) if cit.source_document else "Doc"
        page_label = f"p.{cit.page_number}" if cit.page_number else ""
        img_name   = f"{doc_label} {page_label}".strip() if page_label else doc_label

        if cit.has_image and cit.image_url:
            if os.path.isfile(cit.image_url):
                image_elements.append(
                    cl.Image(
                        name=img_name,
                        path=cit.image_url,
                        display="inline",   # shows as thumbnail in message
                        size="small",       # compact — click to expand full-size
                    )
                )
                # Build a reference line so the image renders in the message body
                image_lines.append(f"[[{img_name}]]")
            elif cit.image_url.startswith("http"):
                image_elements.append(
                    cl.Image(
                        name=img_name,
                        url=cit.image_url,
                        display="inline",
                        size="small",
                    )
                )
                image_lines.append(f"[[{img_name}]]")

    # ── Send image message (if we have images) ────────────────────────────────
    if image_elements:
        # Build a compact header + inline image references
        image_body = "📄 **Document citations** — click to expand full size\n\n"
        image_body += "  ".join(image_lines)
        await cl.Message(
            content=image_body,
            elements=image_elements,
        ).send()

    # ── Build source-list message ─────────────────────────────────────────────
    text_elements: list[cl.Text] = []
    source_lines: list[str] = [f"📎 **{len(citations)} source(s) retrieved**\n"]

    for i, cit in enumerate(citations):
        doc_label     = os.path.basename(cit.source_document) if cit.source_document else "Document"
        page_label    = f"p.{cit.page_number}" if cit.page_number else ""
        section_disp  = _fmt_section(cit.section or "")
        dense_score   = getattr(cit, "dense_score", 0.0) or 0.0
        rrf_score     = cit.confidence or 0.0
        rel_chip      = _relevance_chip(dense_score, rrf_score)

        source_name = f"Source {i + 1}: {doc_label} {page_label}".strip()

        # Compact source line
        meta_parts: list[str] = []
        if page_label:
            meta_parts.append(page_label)
        if section_disp:
            meta_parts.append(section_disp)
        if rel_chip != "—":
            meta_parts.append(rel_chip)

        source_line = f"- [[{source_name}]]"
        if meta_parts:
            source_line += f"  ·  {'  ·  '.join(meta_parts)}"
        source_lines.append(source_line)

        # Text side panel: metadata + excerpt
        preview = (cit.chunk_preview or cit.text or "").strip()
        preview = " ".join(preview.split())
        if len(preview) > 600:
            preview = preview[:597] + "…"

        panel_parts: list[str] = [f"### 📄 {doc_label}", ""]

        rows: list[str] = []
        if cit.page_number:
            rows.append(f"📖 **Page** {cit.page_number}")
        if section_disp:
            rows.append(f"🗂 **Section** {section_disp}")
        rows.append(f"🎯 **Relevance** {rel_chip}")
        panel_parts.append("  ·  ".join(rows))
        panel_parts.append("")

        if preview:
            panel_parts.append("---")
            panel_parts.append("**Excerpt:**")
            panel_parts.append("")
            panel_parts.append(f"> {preview}")
            panel_parts.append("")

        text_elements.append(
            cl.Text(
                name=source_name,
                content="\n".join(panel_parts),
                display="side",
            )
        )

    source_lines.append("")
    await cl.Message(
        content="\n".join(source_lines),
        elements=text_elements,
        actions=[
            cl.Action(
                name="open_sources_sidebar",
                label="📚 View Sources",
                payload={"count": len(citations)},
            )
        ],
    ).send()
