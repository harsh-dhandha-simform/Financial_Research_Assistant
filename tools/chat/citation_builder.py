"""
Citation Builder — transforms LangChain Documents into rich Citation objects.
"""

from langchain_core.documents import Document
from schemas.citation import Citation
from langfuse.decorators import observe

@observe()
def build_citations(docs: list[Document], session_id: str = "") -> list[Citation]:
    """Convert retrieved LangChain Documents into rich Citation objects.
    
    Extracts metadata like page numbers, section filters, and image associations
    that were stored in Qdrant during ingestion. Generates cropped images of the
    chunk location for PDF files.
    """
    from ingestion.image_extractor import extract_chunk_crop
    from sessions.session_store import session_store

    citations = []

    # Pre-fetch user's documents to find local paths for PDFs
    # Use user_id to find docs (user-scoped, not session-scoped)
    doc_paths = {}
    if session_id:
        # session_id is actually the thread_id; look up the session to get user_id
        session = session_store.get(session_id)
        user_id = session.user_identifier if session else ""
        if user_id:
            user_docs = session_store.get_docs_for_user(user_id)
            for d in user_docs:
                if d.get("type") == "pdf" and d.get("local_path"):
                    doc_paths[d.get("name")] = d.get("local_path")
    
    for doc in docs:
        metadata = doc.metadata
        
        # Core fields — try multiple metadata keys with robust fallbacks
        source_doc = (
            metadata.get("source_document")
            or metadata.get("source")
            or metadata.get("file_name")
            or metadata.get("doc_id", "Unknown Document")
        )
        page_num = metadata.get("page", 0)
        section = metadata.get("section", "General")
        
        # Fallback for chunk_id if the retriever didn't pass the raw point ID
        chunk_id = metadata.get("chunk_id", str(hash(doc.page_content)))
        
        # Content fields:
        #   full_text = parent content (used by LLM for broader context, spans pages)
        #   child_text = child content (the actual retrieval unit, ~500 tokens,
        #                precise to a single page — used for crop matching)
        full_text = doc.page_content.strip()
        child_text = metadata.get("child_content", "").strip() or full_text
        text_excerpt = child_text[:200]
        chunk_preview = child_text[:350].replace("\n", " ")
        
        # Relevance scores — use dense_score (cosine similarity, 0–1) for display;
        # RRF scores (0.015–0.016 range) are meaningless to end users
        rrf_score = metadata.get("score", 0.0)
        dense_score = metadata.get("dense_score", 0.0)
        confidence = dense_score if dense_score and dense_score > 0 else rrf_score
        
        # Image fields
        image_url = None
        image_description = None
        image_type = None
        
        # First check if there's a pre-extracted image from Jina
        image_urls = metadata.get("image_urls", [])
        image_descs = metadata.get("image_descriptions", [])
        if len(image_urls) > 0:
            image_url = image_urls[0]
            image_description = image_descs[0] if len(image_descs) > 0 else None
            image_type = "other"
            
        # If no image but we have a local PDF path, generate a crop of the chunk
        elif session_id and page_num > 0 and source_doc and source_doc.lower() in {k.lower(): v for k, v in doc_paths.items()}:
            import os
            # Use case-insensitive lookup
            local_path = next(v for k, v in doc_paths.items() if k.lower() == source_doc.lower())
            if not os.path.isabs(local_path):
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                local_path = os.path.join(project_root, local_path)
            # Use child_text for crop matching — it's more focused and maps
            # precisely to the chunk's page, unlike parent_content which spans pages
            crop_path = extract_chunk_crop(
                local_path, page_num, child_text, session_id, chunk_id=chunk_id
            )
            if crop_path:
                image_url = crop_path
                image_type = "pdf_crop"
                
        has_image = bool(image_url)
        
        citation = Citation(
            text=text_excerpt,
            source_document=source_doc,
            page_number=page_num,
            section=section,
            chunk_id=chunk_id,
            confidence=confidence,
            dense_score=dense_score,
            chunk_preview=chunk_preview,
            has_image=has_image,
            image_url=image_url,
            image_description=image_description,
            image_type=image_type,
        )
        
        citations.append(citation)
        
    return citations
