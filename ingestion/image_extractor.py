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
