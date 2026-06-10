"""
Citation Builder — transforms LangChain Documents into rich Citation objects.
"""

from langchain_core.documents import Document
from schemas.citation import Citation
from langfuse.decorators import observe

@observe()
def build_citations(docs: list[Document]) -> list[Citation]:
    """Convert retrieved LangChain Documents into rich Citation objects.
    
    Extracts metadata like page numbers, section filters, and image associations
    that were stored in Qdrant during ingestion.
    """
    citations = []
    
    for doc in docs:
        metadata = doc.metadata
        
        # Core fields
        source_doc = metadata.get("source_document", metadata.get("source", "Unknown Document"))
        page_num = metadata.get("page", 0)
        section = metadata.get("section", "General")
        
        # Fallback for chunk_id if the retriever didn't pass the raw point ID
        chunk_id = metadata.get("chunk_id", str(hash(doc.page_content)))
        
        # Truncate content for display and strict text extract
        full_text = doc.page_content.strip()
        text_excerpt = full_text[:200]
        chunk_preview = full_text[:150].replace("\n", " ")
        
        # Relevance score (if provided by HybridRetriever)
        confidence = metadata.get("score", 0.0)
        
        # Image fields (populated by Jina ingestion / PyMuPDF)
        image_urls = metadata.get("image_urls", [])
        image_descs = metadata.get("image_descriptions", [])
        
        has_image = len(image_urls) > 0
        image_url = image_urls[0] if has_image else None
        image_description = image_descs[0] if len(image_descs) > 0 else None
        
        citation = Citation(
            text=text_excerpt,
            source_document=source_doc,
            page_number=page_num,
            section=section,
            chunk_id=chunk_id,
            confidence=confidence,
            chunk_preview=chunk_preview,
            has_image=has_image,
            image_url=image_url,
            image_description=image_description,
            image_type="other" if has_image else None
        )
        
        citations.append(citation)
        
    return citations
