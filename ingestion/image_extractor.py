"""
Image extractor — extract and crop PDF content for source citations.

Uses PyMuPDF for:
  - Extracting embedded images from PDFs
  - Rendering full page images
  - Generating precise crop images for text chunks (including tables)

Crop strategy (extract_chunk_crop):
  1. Try find_tables() to detect table bounding boxes — if the chunk's text
     originates from a table, the table rect is used directly for the crop.
  2. Fall back to word-level search_for + block matching — finds the EXACT
     bounding rect (x0, y0, x1, y1) of the matched text, not just Y-strips.
  3. Final fallback: full-page render.

Results are cached in-memory by (pdf_path, page_num, chunk_id_hash).
"""

import logging
import os
import re
import uuid
from typing import Optional

import fitz  # PyMuPDF
from langfuse.decorators import observe

logger = logging.getLogger(__name__)

# Directory where extracted images are stored
IMAGES_BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "images")

# In-memory crop cache: (pdf_path, page_num, chunk_id_hash) → saved_path
_crop_cache: dict[tuple, str] = {}


# ── Embedded image extraction ─────────────────────────────────────────────────

@observe()
def extract_images_from_pdf(
    pdf_path: str,
    session_id: str,
    min_width: int = 100,
    min_height: int = 100,
) -> list[dict]:
    """Extract embedded images from a PDF file."""
    session_dir = os.path.join(IMAGES_BASE_DIR, session_id[:8])
    os.makedirs(session_dir, exist_ok=True)
    extracted: list[dict] = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        logger.error("Failed to open PDF for image extraction: %s", exc)
        return extracted

    for page_num in range(len(doc)):
        page = doc[page_num]
        for img_index, img_info in enumerate(page.get_images(full=True)):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                if not base_image:
                    continue
                image_bytes = base_image["image"]
                ext = base_image.get("ext", "png")
                width = base_image.get("width", 0)
                height = base_image.get("height", 0)
                if width < min_width or height < min_height:
                    continue
                filename = f"page{page_num + 1}_img{img_index + 1}.{ext}"
                save_path = os.path.join(session_dir, filename)
                with open(save_path, "wb") as f:
                    f.write(image_bytes)
                extracted.append({
                    "path": save_path, "page": page_num + 1,
                    "position_index": img_index, "type": "embedded_pdf_image",
                    "width": width, "height": height, "ext": ext,
                })
            except Exception as exc:
                logger.warning("Failed to extract image xref=%d from page %d: %s", xref, page_num + 1, exc)

    doc.close()
    logger.info("Extracted %d images from PDF %s (session: %s)", len(extracted), os.path.basename(pdf_path), session_id[:8])
    return extracted


def extract_images_from_bytes(
    file_bytes: bytes,
    session_id: str,
    min_width: int = 100,
    min_height: int = 100,
) -> list[dict]:
    """Extract images from PDF bytes."""
    import tempfile
    session_dir = os.path.join(IMAGES_BASE_DIR, session_id[:8])
    os.makedirs(session_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, dir=session_dir) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        return extract_images_from_pdf(pdf_path=tmp_path, session_id=session_id,
                                        min_width=min_width, min_height=min_height)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def extract_page_image(
    pdf_path: str,
    page_num: int,
    session_id: str,
    zoom_x: float = 2.0,
    zoom_y: float = 2.0,
) -> Optional[str]:
    """Render a specific PDF page as an image (full page)."""
    session_dir = os.path.join(IMAGES_BASE_DIR, session_id[:8])
    os.makedirs(session_dir, exist_ok=True)
    try:
        doc = fitz.open(pdf_path)
        if page_num < 1 or page_num > len(doc):
            return None
        page = doc[page_num - 1]
        mat = fitz.Matrix(zoom_x, zoom_y)
        pix = page.get_pixmap(matrix=mat)
        filename = f"render_page{page_num}.png"
        save_path = os.path.join(session_dir, filename)
        pix.save(save_path)
        doc.close()
        return save_path
    except Exception as exc:
        logger.warning("Failed to render page %d of %s: %s", page_num, pdf_path, exc)
        return None


# ── Crop helpers ──────────────────────────────────────────────────────────────

def _chunk_looks_like_table(text: str) -> bool:
    """Heuristic: does this chunk contain tabular data?"""
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if len(lines) < 3:
        return False
    num_lines_with_numbers = sum(1 for ln in lines if any(c.isdigit() for c in ln))
    return (num_lines_with_numbers / len(lines)) > 0.5


def _pad_rect(r: fitz.Rect, page: fitz.Page, pad: int) -> fitz.Rect:
    """Return r padded by `pad` pts on all sides, clamped to page bounds."""
    return fitz.Rect(
        max(0, r.x0 - pad),
        max(0, r.y0 - pad),
        min(page.rect.x1, r.x1 + pad),
        min(page.rect.y1, r.y1 + pad),
    )


def _find_table_crop_rect(
    page: fitz.Page,
    text_chunk: str,
    padding: int = 20,
) -> Optional[fitz.Rect]:
    """
    Locate the table on the page that best matches the chunk's numeric content.

    Uses PyMuPDF's find_tables() → scores each table by how many distinct
    numbers from the chunk appear in the table cells.
    """
    try:
        tab_finder = page.find_tables()
        if not tab_finder.tables:
            return None

        chunk_numbers = set(re.findall(r"\d[\d,\.]+", text_chunk))

        if not chunk_numbers:
            # No numbers — pick largest table
            best = max(tab_finder.tables, key=lambda t: (t.bbox[2] - t.bbox[0]) * (t.bbox[3] - t.bbox[1]))
            return _pad_rect(fitz.Rect(best.bbox), page, padding)

        best_table = None
        best_score = 0

        for table in tab_finder.tables:
            try:
                table_text = " ".join(str(cell) for row in table.extract() for cell in row if cell)
            except Exception:
                table_text = ""
            score = sum(1 for num in chunk_numbers if num in table_text)
            if score > best_score:
                best_score = score
                best_table = table

        if best_table is None or best_score == 0:
            return None

        return _pad_rect(fitz.Rect(best_table.bbox), page, padding)

    except Exception as exc:
        logger.debug("Table detection failed: %s", exc)
        return None


def _find_text_crop_rect(
    page: fitz.Page,
    text_chunk: str,
    padding: int = 30,
) -> Optional[fitz.Rect]:
    """
    Find the tight bounding rect of a chunk's text on the page.

    Algorithm:
      1. Extract clean anchor phrases (first 3 sentences/segments of the chunk,
         each trimmed to ≤40 chars so search_for finds exact matches).
      2. Use page.search_for() to get word-level quads (precise x0/y0/x1/y1).
      3. Combine all matched rects into a single tight bounding box.
      4. Also fall back to block-level matching for any anchor not found by search_for.

    Returns the padded bounding Rect, or None if nothing matched.
    """
    clean_text = " ".join(text_chunk.split())  # collapse whitespace

    # Build anchors: take phrases every ~100 chars, trimmed to ≤40 chars
    anchors: list[str] = []
    for i in range(0, min(len(clean_text), 400), 80):
        segment = clean_text[i:i + 80].strip()
        # Trim at last word boundary to ≤40 chars
        anchor = segment[:45]
        # Don't stop mid-word
        if len(segment) > 45 and " " in anchor:
            anchor = anchor[:anchor.rfind(" ")]
        anchor = anchor.strip()
        if len(anchor) >= 6 and anchor not in anchors:
            anchors.append(anchor)
        if len(anchors) >= 4:
            break

    if not anchors:
        return None

    matched_rects: list[fitz.Rect] = []

    # Strategy A: word-level search_for (most precise)
    for anchor in anchors:
        try:
            hits = page.search_for(anchor, quads=False)
            matched_rects.extend(hits)
        except Exception:
            pass

    # Strategy B: block-level fallback for any anchor not found above
    if len(matched_rects) < 2:
        blocks = page.get_text("blocks")  # [(x0,y0,x1,y1,text,…)]
        for anchor in anchors:
            anchor_lower = anchor.lower()
            for block in blocks:
                x0, y0, x1, y1, block_text, *_ = block
                if len(block_text.strip()) < 4:
                    continue
                if anchor_lower in block_text.lower():
                    matched_rects.append(fitz.Rect(x0, y0, x1, y1))
                    break

    if not matched_rects:
        return None

    # Compute tight bounding box over all matched rects
    x0 = min(r.x0 for r in matched_rects)
    y0 = min(r.y0 for r in matched_rects)
    x1 = max(r.x1 for r in matched_rects)
    y1 = max(r.y1 for r in matched_rects)

    crop_rect = fitz.Rect(x0, y0, x1, y1)
    if crop_rect.width < 20 or crop_rect.height < 10:
        return None

    return _pad_rect(crop_rect, page, padding)


def _render_crop(
    page: fitz.Page,
    crop_rect: fitz.Rect,
    zoom: float = 2.5,
) -> Optional[fitz.Pixmap]:
    """Render a cropped region of a page at the given zoom level."""
    try:
        mat = fitz.Matrix(zoom, zoom)
        return page.get_pixmap(matrix=mat, clip=crop_rect)
    except Exception as exc:
        logger.debug("Render crop failed: %s", exc)
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def extract_chunk_crop(
    pdf_path: str,
    page_num: int,
    text_chunk: str,
    session_id: str,
    chunk_id: str = "",
    zoom_x: float = 2.5,
    zoom_y: float = 2.5,
    padding: int = 25,
) -> Optional[str]:
    """
    Render a precise cropped image of the chunk's location on a PDF page.

    Strategy (in preference order):
      1. Table detection — if the chunk looks tabular, locate the matching table.
      2. Text search — use page.search_for() + block matching to find the exact
         bounding box (full x0/y0/x1/y1, not just a horizontal strip).
      3. Full-page fallback — render the entire page.

    Args:
        pdf_path:   Absolute path to the PDF.
        page_num:   1-indexed page number.
        text_chunk: The chunk text to locate on the page.
        session_id: Session ID for the image directory.
        chunk_id:   Optional chunk ID for cache keying.
        zoom_x/y:   Render resolution (2.5× gives ~200 DPI for A4).
        padding:    Points of whitespace context around the matched region.

    Returns:
        Absolute path to the saved PNG, or None on failure.
    """
    if not text_chunk or len(text_chunk.strip()) < 10:
        return None

    # Cache check
    cache_key = (pdf_path, page_num, chunk_id or hash(text_chunk[:200]))
    if cache_key in _crop_cache:
        cached = _crop_cache[cache_key]
        if os.path.isfile(cached):
            logger.debug("Crop cache hit: page %d of %s", page_num, os.path.basename(pdf_path))
            return cached

    session_dir = os.path.join(IMAGES_BASE_DIR, session_id[:8])
    os.makedirs(session_dir, exist_ok=True)

    try:
        doc = fitz.open(pdf_path)
        if page_num < 1 or page_num > len(doc):
            doc.close()
            return None

        page = doc[page_num - 1]

        crop_rect: Optional[fitz.Rect] = None
        method = "unknown"

        # ── Strategy 1: Table detection ───────────────────────────────────────
        if _chunk_looks_like_table(text_chunk):
            crop_rect = _find_table_crop_rect(page, text_chunk, padding=padding)
            if crop_rect:
                method = "table"
                logger.debug("Crop method=table: page %d rect=%s", page_num, crop_rect)

        # ── Strategy 2: Text search ───────────────────────────────────────────
        if crop_rect is None:
            crop_rect = _find_text_crop_rect(page, text_chunk, padding=padding)
            if crop_rect:
                method = "text_search"
                logger.debug("Crop method=text_search: page %d rect=%s", page_num, crop_rect)

        # ── Strategy 3: Full-page fallback ────────────────────────────────────
        if crop_rect is None:
            logger.debug("Crop: no match on page %d — full page fallback", page_num)
            doc.close()
            result = extract_page_image(pdf_path, page_num, session_id, zoom_x, zoom_y)
            if result:
                _crop_cache[cache_key] = result
            return result

        # ── Render the matched region ─────────────────────────────────────────
        mat = fitz.Matrix(zoom_x, zoom_y)
        pix = page.get_pixmap(matrix=mat, clip=crop_rect)

        filename = f"crop_p{page_num}_{uuid.uuid4().hex[:6]}.png"
        save_path = os.path.join(session_dir, filename)
        pix.save(save_path)
        doc.close()

        _crop_cache[cache_key] = save_path
        logger.info("Crop saved [%s]: page %d → %s (%.0f×%.0f pts)",
                    method, page_num, filename, crop_rect.width, crop_rect.height)
        return save_path

    except Exception as exc:
        logger.warning("extract_chunk_crop failed page %d of %s: %s",
                       page_num, os.path.basename(pdf_path), exc)
        return None
