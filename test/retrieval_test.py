import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging
logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')

from retrieval import QdrantStore, BM25Retriever, HybridRetriever
from schemas.chunks import ChildChunk, ParentChunk

# ── Setup test data ──
print('── Setting up test data ──')
parent_mda = ParentChunk(
    doc_id='doc-001', content='Management Discussion: Apple FY2024 financial overview.',
    chunk_index=0, token_count=100, section='mda',
)
parent_risk = ParentChunk(
    doc_id='doc-001', content='Risk Factors section discussing competition and regulation.',
    chunk_index=1, token_count=80, section='risk_factors',
)

children = [
    ChildChunk(parent_id=parent_mda.chunk_id, doc_id='doc-001',
               content='Apple Inc. (AAPL) reported total revenue of \$394.3 billion for fiscal year 2024, an increase of 3% year-over-year.',
               chunk_index=0, token_count=25, section='mda'),
    ChildChunk(parent_id=parent_mda.chunk_id, doc_id='doc-001',
               content='Gross margin was 46.2% for FY2024 compared to 44.1% for FY2023. EBITDA margin expanded by 150 basis points.',
               chunk_index=1, token_count=22, section='mda'),
    ChildChunk(parent_id=parent_mda.chunk_id, doc_id='doc-001',
               content='Services revenue reached \$99.3 billion, driven by App Store, iCloud, and Apple Music subscription growth.',
               chunk_index=2, token_count=20, section='mda'),
    ChildChunk(parent_id=parent_mda.chunk_id, doc_id='doc-001',
               content='Net income was \$93.7 billion with diluted EPS of \$6.08, compared to \$6.16 in the prior year.',
               chunk_index=3, token_count=22, section='mda'),
    ChildChunk(parent_id=parent_risk.chunk_id, doc_id='doc-001',
               content='The Company faces substantial competition from Samsung, Google, and Huawei in the smartphone market.',
               chunk_index=4, token_count=18, section='risk_factors'),
    ChildChunk(parent_id=parent_risk.chunk_id, doc_id='doc-001',
               content='Regulatory changes in the EU Digital Markets Act may require significant changes to App Store policies.',
               chunk_index=5, token_count=18, section='risk_factors'),
]

parent_lookup = {
    parent_mda.chunk_id: parent_mda,
    parent_risk.chunk_id: parent_risk,
}

# ── 1. Setup Qdrant + BM25 ──
store = QdrantStore(collection_name='test_hybrid')
store.upsert_chunks(children, parent_lookup)

bm25 = BM25Retriever()
bm25.build_from_child_chunks(children, parent_lookup)

print(f'  ✅ Qdrant: {store.collection_info()}')
print(f'  ✅ BM25: {len(children)} chunks indexed')

# ── 2. Dense-only search ──
print()
print('── Dense-Only Search: \"revenue growth\" ──')
dense = store.search('revenue growth', top_k=3)
for r in dense:
    print(f'  [{r.score:.4f}] [{r.section:15s}] {r.content[:65]}...')

# ── 3. BM25-only search ──
print()
print('── BM25-Only Search: \"EBITDA margin\" ──')
sparse = bm25.search('EBITDA margin', top_k=3)
for r in sparse:
    print(f'  [{r.score:.4f}] [{r.section:15s}] {r.content[:65]}...')

# ── 4. BM25 with financial terms (where it shines) ──
print()
print('── BM25 Search: \"AAPL EPS diluted\" (exact terms) ──')
sparse2 = bm25.search('AAPL EPS diluted', top_k=3)
for r in sparse2:
    print(f'  [{r.score:.4f}] [{r.section:15s}] {r.content[:65]}...')

# ── 5. HYBRID search ──
print()
print('── Hybrid Search (Dense + BM25 + RRF): \"Apple AAPL revenue growth\" ──')
hybrid = HybridRetriever(store, bm25)
result = hybrid.search('Apple AAPL revenue growth', top_k=4)
for c in result.chunks:
    print(f'  RRF={c.rrf_score:.4f} dense={c.dense_score:.4f} sparse={c.sparse_score:.4f} [{c.section}]')
    print(f'    {c.content[:70]}...')

# ── 6. Context text (what agents receive) ──
print()
print('── Context Text for LLM Prompt ──')
print(result.context_text[:400])
print('...')

# ── 7. Section-filtered hybrid ──
print()
result2 = hybrid.search('competition Samsung market', top_k=3, section_filter='risk_factors')
print(f'── Hybrid + section filter (risk_factors) → {len(result2.chunks)} results ──')
for c in result2.chunks:
    print(f'  RRF={c.rrf_score:.4f} [{c.section}] {c.content[:60]}...')

# Cleanup
store.delete_collection()
print()
print('=' * 60)
print('  ✅  Module 7 complete — Hybrid retrieval + RRF working.')
print('=' * 60)

