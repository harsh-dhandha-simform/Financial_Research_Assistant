"""
Qdrant vector store — collection management, upsert, and dense search.

Stores child chunk embeddings in a Qdrant collection for dense vector
retrieval. Parent chunk text is stored in the payload for context
expansion at query time.

Data flow:
    ChildChunks → embed_texts() → Qdrant upsert (vectors + payload)
    Query       → embed_query() → Qdrant search → RetrievedChunks

Usage:
    from retrieval.qdrant_store import QdrantStore

    store = QdrantStore()
    store.upsert_chunks(child_chunks, parent_lookup)
    results = store.search("Apple revenue growth", top_k=10)
"""

import logging
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from config import settings
from retrieval.embedding import (
    embed_query,
    embed_texts,
    get_embedding_dimensions,
    get_embedding_model,
)
from schemas.chunks import ChildChunk, ParentChunk
from schemas.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "financial_chunks"


class QdrantStore:
    """Manages a Qdrant collection for child chunk embeddings.

    Each point in Qdrant stores:
      - vector: dense embedding of the child chunk text
      - payload: chunk_id, parent_id, doc_id, section, content,
                 parent_content (for context expansion), and metadata
    """

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION,
    ):
        self.collection_name = collection_name
        self.dimensions = get_embedding_dimensions()

        # Connect to Qdrant
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
        logger.info(
            "Connected to Qdrant at %s (collection: %s, dims: %d)",
            settings.qdrant_url,
            collection_name,
            self.dimensions,
        )

    def ensure_collection(self) -> None:
        """Create the collection if it doesn't exist.

        Configures:
          - Cosine similarity (for normalized embeddings)
          - HNSW indexing (m=16, ef_construct=100) for fast ANN search
          - Payload indexes on section and doc_id for filtered queries
        """
        collections = [c.name for c in self.client.get_collections().collections]

        if self.collection_name in collections:
            logger.info("Collection '%s' already exists", self.collection_name)
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.dimensions,
                distance=Distance.COSINE,
            ),
            hnsw_config=HnswConfigDiff(
                m=16,                # Number of edges per node (default 16)
                ef_construct=100,    # Construction-time search width (higher = better recall)
            ),
        )
        logger.info(
            "Created collection '%s' (size=%d, COSINE, HNSW m=16 ef=100)",
            self.collection_name,
            self.dimensions,
        )

        # Payload indexes for fast filtered search
        for field_name in ("section", "doc_id"):
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        logger.info("Created payload indexes on 'section' and 'doc_id'")

    def upsert_chunks(
        self,
        children: list[ChildChunk],
        parent_lookup: dict[str, ParentChunk],
        batch_size: int = 64,
    ) -> int:
        """Embed child chunks and upsert them into Qdrant.

        Each child's payload includes the parent chunk content so the
        retriever can expand context without a separate lookup.

        Args:
            children: Child chunks to embed and store.
            parent_lookup: Mapping of parent_id → ParentChunk for context.
            batch_size: Number of chunks to embed and upsert at a time.

        Returns:
            Number of points upserted.
        """
        if not children:
            return 0

        self.ensure_collection()
        model = get_embedding_model()
        total_upserted = 0

        for i in range(0, len(children), batch_size):
            batch = children[i : i + batch_size]

            # Embed the batch
            texts = [child.content for child in batch]
            vectors = embed_texts(texts, model)

            # Build Qdrant points
            points: list[PointStruct] = []
            for child, vector in zip(batch, vectors):
                parent = parent_lookup.get(child.parent_id)
                parent_content = parent.content if parent else ""

                point = PointStruct(
                    id=child.chunk_id,
                    vector=vector,
                    payload={
                        "chunk_id": child.chunk_id,
                        "parent_id": child.parent_id,
                        "doc_id": child.doc_id,
                        "section": child.section,
                        "content": child.content,
                        "parent_content": parent_content,
                        "chunk_index": child.chunk_index,
                        "token_count": child.token_count,
                        "metadata": child.metadata,
                    },
                )
                points.append(point)

            # Upsert batch
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            total_upserted += len(points)
            logger.info(
                "Upserted batch %d-%d (%d points)",
                i,
                i + len(batch),
                len(points),
            )

        logger.info(
            "Total upserted: %d child chunks into '%s'",
            total_upserted,
            self.collection_name,
        )
        return total_upserted

    def search(
        self,
        query: str,
        top_k: int = 10,
        section_filter: str | None = None,
        doc_id_filter: str | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        """Dense vector search over child chunk embeddings.

        Args:
            query: Natural language search query.
            top_k: Number of results to return.
            section_filter: Only return chunks from this section
                            (e.g. 'risk_factors', 'mda').
            doc_id_filter: Only return chunks from this document.
            score_threshold: Minimum similarity score (0-1).

        Returns:
            List of RetrievedChunk objects sorted by relevance.
        """
        query_vector = embed_query(query)

        # Build optional filters
        must_conditions = []
        if section_filter:
            must_conditions.append(
                FieldCondition(key="section", match=MatchValue(value=section_filter))
            )
        if doc_id_filter:
            must_conditions.append(
                FieldCondition(key="doc_id", match=MatchValue(value=doc_id_filter))
            )
        query_filter = Filter(must=must_conditions) if must_conditions else None

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )

        # Convert to RetrievedChunk objects
        retrieved: list[RetrievedChunk] = []
        for point in results.points:
            payload = point.payload or {}
            chunk = RetrievedChunk(
                chunk_id=payload.get("chunk_id", ""),
                content=payload.get("content", ""),
                parent_content=payload.get("parent_content", ""),
                doc_id=payload.get("doc_id", ""),
                section=payload.get("section", ""),
                score=point.score if point.score is not None else 0.0,
                retrieval_method="dense",
            )
            retrieved.append(chunk)

        logger.info(
            "Dense search '%s' → %d results (section=%s, doc=%s)",
            query[:50],
            len(retrieved),
            section_filter or "all",
            doc_id_filter[:8] if doc_id_filter else "all",
        )
        return retrieved

    def delete_collection(self) -> None:
        """Delete the collection (useful for testing / reset)."""
        self.client.delete_collection(self.collection_name)
        logger.info("Deleted collection '%s'", self.collection_name)

    def collection_info(self) -> dict:
        """Get collection statistics."""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "points_count": info.points_count,
            "status": info.status.value,
        }
