
import logging
logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')

from retrieval.embedding import get_embedding_model, embed_query, embed_texts
from retrieval.embedding import get_embedding_dimensions, get_active_model_name, EMBEDDING_CHAIN

print('── Embedding Fallback Chain (Inference APIs) ──')
for i, (name, provider, model_id, dims) in enumerate(EMBEDDING_CHAIN, 1):
    print(f'  {i}. {name:35s} [{provider:12s}] {dims} dims')

print()
print('Initializing...')
model = get_embedding_model()
print(f'  ✅ Active model: {get_active_model_name()}')
print(f'  ✅ Dimensions:   {get_embedding_dimensions()}')

# Test financial queries
print()
print('── Financial Queries ──')
queries = [
    'Apple AAPL revenue for FY2024',
    'EBITDA margin expanded 150 basis points',
    'Risk factors competition smartphone market',
]
for q in queries:
    vec = embed_query(q)
    print(f'  ✅ [{len(vec)} dims] {q}')

vectors = embed_texts(queries)
print(f'  ✅ Batch: {len(vectors)} texts → {len(vectors[0])}-dim each')

print()
print('=' * 60)
print('  ✅  Inference API embedding chain working.')
print('=' * 60)

