import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')

from retrieval.embedding import _try_openai_embeddings

if __name__ == "__main__":
    print("Testing OpenAI embeddings...")
    model_id = "text-embedding-3-large"
    model = _try_openai_embeddings(model_id)
    
    if model is not None:
        print(f"✅ Successfully initialized {model_id}")
        test_vec = model.embed_query("test query")
        print(f"✅ Dimensions: {len(test_vec)}")
    else:
        print("❌ Failed to initialize or API_KEY not set")
