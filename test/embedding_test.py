import logging
logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')

from retrieval import QdrantStore, embed_texts, embed_query, get_embedding_model
from retrieval.embedding import get_embedding_dimensions, get_active_model_name
from schemas.chunks import ChildChunk, ParentChunk

# Test 1: Embedding fallback chain
print('── Embedding Fallback Chain ──')
model = get_embedding_model()
print(f'  ✅ Active model: {get_active_model_name()}')
print(f'  ✅ Dimensions:   {get_embedding_dimensions()}')

vec = embed_query('What is Apple revenue?')
print(f'  ✅ Query embed:  {len(vec)} dims')

vectors = embed_texts(['Apple revenue', 'Net income', 'Competition risk'])
print(f'  ✅ Batch embed:  {len(vectors)} texts, {len(vectors[0])}-dim each')

# Test 2: Qdrant store with HNSW
print()
print('── Qdrant Store (HNSW) ──')
store = QdrantStore(collection_name='test_hnsw_chunks')

parent = ParentChunk(
    doc_id='doc-001', content='Apple financial overview for FY2024.',
    chunk_index=0, token_count=100, section='mda',
)
children = [
    ChildChunk(parent_id=parent.chunk_id, doc_id='doc-001',
               content='Apple revenue was 394.3 billion for fiscal year 2024.',
               chunk_index=0, token_count=15, section='mda'),
    ChildChunk(parent_id=parent.chunk_id, doc_id='doc-001',
               content='Net income was 93.7 billion, a decrease of 3 percent.',
               chunk_index=1, token_count=14, section='mda'),
    ChildChunk(parent_id=parent.chunk_id, doc_id='doc-001',
               content='Competition in smartphones remains a key business risk.',
               chunk_index=2, token_count=10, section='risk_factors'),
]
parent_lookup = {parent.chunk_id: parent}

count = store.upsert_chunks(children, parent_lookup)
print(f'  ✅ Upserted: {count} chunks')
info = store.collection_info()
print(f'  ✅ Collection: {info}')

# Dense search
print()
print('── Dense Search Results ──')
results = store.search('What was Apple revenue?', top_k=3)
for r in results:
    print(f'  ✅ [{r.score:.4f}] [{r.section:15s}] {r.content[:60]}')

# Section filter
print()
results = store.search('business risk', top_k=3, section_filter='risk_factors')
print(f'── Section filter (risk_factors) → {len(results)} results ──')
for r in results:
    print(f'  ✅ [{r.score:.4f}] [{r.section}] {r.content[:60]}')

# Cleanup
store.delete_collection()
print(f'  ✅ Test collection deleted')

print()
print('=' * 60)
print('  ✅  Module 6 complete — Qdrant + HNSW + embeddings.')
print('=' * 60)
