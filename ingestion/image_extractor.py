"""
Image extractor — extract embedded images from PDF documents using PyMuPDF.

Extracts images at ingestion time and stores them locally. No vision
description happens here — that is done lazily at retrieval time by
image_describer.py.

Usage:
    from ingestion.image_extractor import extract_images_from_pdf

    images = extract_images_from_pdf("path/to/file.pdf", session_id="abc12345")
    # returns [{path, page, position_index, type, width, height, ext}]
"""

import logging
import os
from typing import Optional

import fitz  # PyMuPDF
from langfuse.decorators import observe

logger = logging.getLogger(__name__)

# Directory where extracted images are stored
IMAGES_BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "images")


@observe()
def extract_images_from_pdf(
    pdf_path: str,
    session_id: str,
    min_width: int = 100,
    min_height: int = 100,
) -> list[dict]:
    """Extract embedded images from a PDF file.

    Only extracts images larger than min_width x min_height to filter
    out icons, bullets, and decorative elements.

    Args:
        pdf_path: Path to the PDF file.
        session_id: Session ID for organizing images by session.
        min_width: Minimum image width in pixels.
        min_height: Minimum image height in pixels.

    Returns:
        List of dicts: {path, page, position_index, type, width, height, ext}
    """
    # Create output directory for this session
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
        image_list = page.get_images(full=True)

        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]

            try:
                base_image = doc.extract_image(xref)
                if not base_image:
                    continue

                image_bytes = base_image["image"]
                ext = base_image.get("ext", "png")
                width = base_image.get("width", 0)
                height = base_image.get("height", 0)

                # Skip small images (icons, bullets, decorative)
                if width < min_width or height < min_height:
                    continue

                # Save image
                filename = f"page{page_num + 1}_img{img_index + 1}.{ext}"
                save_path = os.path.join(session_dir, filename)

                with open(save_path, "wb") as f:
                    f.write(image_bytes)

                extracted.append({
                    "path": save_path,
                    "page": page_num + 1,
                    "position_index": img_index,
                    "type": "embedded_pdf_image",
                    "width": width,
                    "height": height,
                    "ext": ext,
                })

            except Exception as exc:
                logger.warning(
                    "Failed to extract image xref=%d from page %d: %s",
                    xref, page_num + 1, exc,
                )

    doc.close()

    logger.info(
        "Extracted %d images from PDF %s (session: %s)",
        len(extracted), os.path.basename(pdf_path), session_id[:8],
    )
    return extracted


def extract_images_from_bytes(
    file_bytes: bytes,
    session_id: str,
    min_width: int = 100,
    min_height: int = 100,
) -> list[dict]:
    """Extract images from PDF bytes (for uploaded files without a path).

    Writes bytes to a temp file, then delegates to extract_images_from_pdf.
    """
    import tempfile

    session_dir = os.path.join(IMAGES_BASE_DIR, session_id[:8])
    os.makedirs(session_dir, exist_ok=True)

    # Write bytes to a temp PDF
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, dir=session_dir) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        return extract_images_from_pdf(
            pdf_path=tmp_path,
            session_id=session_id,
            min_width=min_width,
            min_height=min_height,
        )
    finally:
        # Clean up temp PDF
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
    """Render a specific PDF page as an image.
    
    Args:
        pdf_path: Path to the PDF file.
        page_num: Page number (1-indexed).
        session_id: Session ID for storing the image.
        zoom_x, zoom_y: Zoom factors for resolution.
        
    Returns:
        Path to the saved image file, or None if failed.
    """
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

def extract_chunk_crop(
    pdf_path: str,
    page_num: int,
    text_chunk: str,
    session_id: str,
    zoom_x: float = 2.0,
    zoom_y: float = 2.0,
) -> Optional[str]:
    """Render a cropped image of the specific text region on a PDF page.
    
    Searches for the text on the page to find its bounding box, pads it,
    and returns a cropped image of just that region.
    """
    if not text_chunk or len(text_chunk) < 10:
        return None
        
    import uuid
    session_dir = os.path.join(IMAGES_BASE_DIR, session_id[:8])
    os.makedirs(session_dir, exist_ok=True)
    
    try:
        doc = fitz.open(pdf_path)
        if page_num < 1 or page_num > len(doc):
            return None
            
        page = doc[page_num - 1]
        
        clean_text = text_chunk.strip()
        lines = [line.strip() for line in clean_text.split("\n") if len(line.strip()) > 10]
        if not lines:
            lines = [clean_text[:50]]
            
        search_lines = [lines[0]]
        if len(lines) > 1:
            search_lines.append(lines[-1]) # last line
        if len(lines) > 2:
            search_lines.append(lines[len(lines)//2]) # middle line
            
        found_rects = []
        for line in search_lines:
            candidate = line[:40] # take up to 40 chars to avoid wrapping issues
            if not candidate:
                continue
            r = page.search_for(candidate)
            if r:
                found_rects.extend(r)
        
        if not found_rects:
            # Fallback to rendering the whole page if text not found
            doc.close()
            return extract_page_image(pdf_path, page_num, session_id, zoom_x, zoom_y)
            
        # Create a bounding box covering the found text, with padding
        rect = found_rects[0]
        for r in found_rects[1:]:
            rect = rect | r  # Union of rectangles
            
        rect.x0 = max(0, rect.x0 - 30)
        rect.y0 = max(0, rect.y0 - 30)
        rect.x1 = min(page.rect.width, rect.x1 + 30)
        rect.y1 = min(page.rect.height, rect.y1 + 40)
        
        mat = fitz.Matrix(zoom_x, zoom_y)
        pix = page.get_pixmap(matrix=mat, clip=rect)
        
        filename = f"crop_p{page_num}_{uuid.uuid4().hex[:6]}.png"
        save_path = os.path.join(session_dir, filename)
        
        pix.save(save_path)
        doc.close()
        return save_path
    except Exception as exc:
        logger.warning("Failed to render crop for page %d of %s: %s", page_num, pdf_path, exc)
        return None
