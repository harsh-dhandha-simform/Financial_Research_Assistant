"""
Chonkie-based chunking with parent-child hierarchy and section detection.

Pipeline:
    RawDocument
        → Section boundary detection (regex on SEC headers)
        → Parent chunking (~2000 tokens via RecursiveChunker)
            → Child chunking (~500 tokens per parent via SentenceChunker)
        → list[ParentChunk], list[ChildChunk]

Chunker choices:
  - RecursiveChunker for parents: splits on \\n\\n (paragraphs) first,
    then sentences, then punctuation. Keeps financial paragraphs and
    table rows intact inside parent chunks.
  - SentenceChunker for children: ensures each retrieval unit holds
    complete sentences, producing better embeddings and coherent context.

Usage:
    from chunking import chunk_document

    parents, children = chunk_document(raw_document)
"""

import logging
import re

from chonkie import RecursiveChunker, SentenceChunker

from schemas.chunks import STANDARD_SECTIONS, ChildChunk, ParentChunk
from schemas.ingestion import RawDocument

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Section detection — maps SEC filing headers to STANDARD_SECTIONS keys
# ═════════════════════════════════════════════════════════════════════════════

# Each tuple: (compiled regex, standard section key)
# Order matters — more specific patterns first to avoid false matches.
_SECTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    # 10-K / 10-Q Item-based headers (most specific first)
    (re.compile(r"(?im)^\s*item\s+1a\.?\s*[\.\:\—\-–]?\s*risk\s+factors?", re.MULTILINE), "risk_factors"),
    (re.compile(r"(?im)^\s*item\s+1\.?\s*[\.\:\—\-–]?\s*business\b", re.MULTILINE), "business"),
    (re.compile(r"(?im)^\s*item\s+7a\.?\s*[\.\:\—\-–]?\s*quantitative", re.MULTILINE), "market_risk"),
    (re.compile(r"(?im)^\s*item\s+7\.?\s*[\.\:\—\-–]?\s*management", re.MULTILINE), "mda"),
    (re.compile(r"(?im)^\s*item\s+8\.?\s*[\.\:\—\-–]?\s*financial\s+statements?", re.MULTILINE), "financial_statements"),
    (re.compile(r"(?im)^\s*item\s+3\.?\s*[\.\:\—\-–]?\s*legal\s+proceedings?", re.MULTILINE), "legal_proceedings"),
    (re.compile(r"(?im)^\s*item\s+2\.?\s*[\.\:\—\-–]?\s*properties?", re.MULTILINE), "properties"),
    # Keyword-based fallbacks (for non-standard headers / Proxy statements)
    (re.compile(r"(?i)\brisk\s+factors?\b"), "risk_factors"),
    (re.compile(r"(?i)\bmanagement.{0,3}s?\s+discussion\s+(?:and|&)\s+analysis\b"), "mda"),
    (re.compile(r"(?i)\bfinancial\s+statements?\s+(?:and\s+)?(?:supplementary|notes)\b"), "financial_statements"),
    (re.compile(r"(?i)\bexecutive\s+compensation\b"), "executive_compensation"),
    (re.compile(r"(?i)\bcorporate\s+governance\b"), "corporate_governance"),
    (re.compile(r"(?i)\bshareholder\s+proposals?\b"), "shareholder_proposals"),
    (re.compile(r"(?i)\blegal\s+proceedings?\b"), "legal_proceedings"),
]


def _detect_sections(text: str) -> list[tuple[int, str]]:
    """Scan the document text for section headers and return their positions.

    Returns:
        Sorted list of (char_position, section_label) tuples.
        Section labels are keys from STANDARD_SECTIONS.
    """
    boundaries: list[tuple[int, str]] = []
    seen_positions: set[int] = set()

    for pattern, label in _SECTION_PATTERNS:
        for match in pattern.finditer(text):
            pos = match.start()
            # Avoid duplicate detections within 100 chars of each other
            if any(abs(pos - sp) < 100 for sp in seen_positions):
                continue
            boundaries.append((pos, label))
            seen_positions.add(pos)

    boundaries.sort(key=lambda x: x[0])
    return boundaries


def _section_at(boundaries: list[tuple[int, str]], char_pos: int) -> str:
    """Determine which section a given character position falls in.

    Walks the sorted boundaries and returns the label of the last
    section header that appears before char_pos.
    """
    current = ""
    for boundary_pos, label in boundaries:
        if boundary_pos > char_pos:
            break
        current = label
    return current


# ═════════════════════════════════════════════════════════════════════════════
# Parent-child chunking pipeline
# ═════════════════════════════════════════════════════════════════════════════


def chunk_document(
    document: RawDocument,
    parent_size: int = 2000,
    parent_overlap: int = 200,
    child_size: int = 500,
    child_overlap: int = 50,
) -> tuple[list[ParentChunk], list[ChildChunk]]:
    """Chunk a RawDocument into a parent-child hierarchy.

    1. Detects SEC section headers in the full text.
    2. Splits into parent chunks (~parent_size tokens) using RecursiveChunker
       (respects paragraph boundaries — keeps financial data intact).
    3. Splits each parent into child chunks (~child_size tokens) using
       SentenceChunker (never splits mid-sentence for better embeddings).
    4. Each child links to its parent; both carry section labels.

    Args:
        document: The RawDocument to chunk.
        parent_size: Target token count for parent chunks.
        parent_overlap: Token overlap between consecutive parents.
        child_size: Target token count for child chunks.
        child_overlap: Token overlap between consecutive children.

    Returns:
        Tuple of (parent_chunks, child_chunks).
    """
    text = document.content
    if not text.strip():
        logger.warning("Empty document content, returning no chunks")
        return [], []

    doc_id = document.doc_id
    # Extract source info for chunk metadata
    source_file_name = document.metadata.file_name or ""
    source_company = document.metadata.company_name or ""

    # ── Step 1: Detect section boundaries ────────────────────────────────────
    section_boundaries = _detect_sections(text)
    if section_boundaries:
        detected = [f"{label}@{pos}" for pos, label in section_boundaries]
        logger.info("Detected %d sections: %s", len(section_boundaries), ", ".join(detected))
    else:
        logger.info("No standard sections detected — all chunks will have section=''")

    # ── Step 2: Parent chunking (RecursiveChunker) ───────────────────────────
    # Splits on \n\n first (paragraphs), then sentences, then punctuation.
    # This keeps financial paragraphs and table rows intact.
    parent_chunker = RecursiveChunker(
        chunk_size=parent_size,
    )
    raw_parents = parent_chunker.chunk(text)

    parents: list[ParentChunk] = []
    children: list[ChildChunk] = []

    # ── Step 3: Child chunking (SentenceChunker) ─────────────────────────────
    # Never splits mid-sentence — each child is a coherent retrieval unit.
    child_chunker = SentenceChunker(
        chunk_size=child_size,
        chunk_overlap=child_overlap,
    )

    for parent_idx, raw_parent in enumerate(raw_parents):
        # Determine section from the midpoint of the parent chunk
        midpoint = raw_parent.start_index + (raw_parent.end_index - raw_parent.start_index) // 2
        section = _section_at(section_boundaries, midpoint)

        parent = ParentChunk(
            doc_id=doc_id,
            content=raw_parent.text,
            chunk_index=parent_idx,
            token_count=raw_parent.token_count,
            section=section,
            metadata={
                "start_index": raw_parent.start_index,
                "end_index": raw_parent.end_index,
            },
        )
        parents.append(parent)

        # Split parent into child chunks
        raw_children = child_chunker.chunk(raw_parent.text)

        for child_idx, raw_child in enumerate(raw_children):
            child = ChildChunk(
                parent_id=parent.chunk_id,
                doc_id=doc_id,
                content=raw_child.text,
                chunk_index=child_idx,
                token_count=raw_child.token_count,
                section=section,
                metadata={
                    "parent_chunk_index": parent_idx,
                    "start_index_in_parent": raw_child.start_index,
                    "end_index_in_parent": raw_child.end_index,
                    "source_document": source_file_name,
                    "company_name": source_company,
                    "page": parent_idx,  # approximate page from parent index
                },
            )
            children.append(child)

    logger.info(
        "Chunked doc %s → %d parents, %d children (avg %.0f tokens/parent, %.0f tokens/child)",
        doc_id[:8],
        len(parents),
        len(children),
        sum(p.token_count for p in parents) / max(len(parents), 1),
        sum(c.token_count for c in children) / max(len(children), 1),
    )

    return parents, children
