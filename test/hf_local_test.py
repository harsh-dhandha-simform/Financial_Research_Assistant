import sys
import os
import time
import logging
from langchain_core.embeddings import Embeddings

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging to show info and warning messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from retrieval.embedding import _try_hf_local_embeddings

if __name__ == "__main__":
    # Get model ID from command line arguments or default to Qwen
    model_id = "Qwen/Qwen3-Embedding-0.6B"
    if len(sys.argv) > 1:
        model_id = sys.argv[1]

    print("=" * 70)
    print("      HuggingFace Local Embeddings Test")
    print("=" * 70)
    print(f"Target Model: {model_id}")
    if model_id == "Qwen/Qwen3-Embedding-0.6B":
        print("NOTE: The first run will download the model (~1.2GB). This may take a few minutes...")
    else:
        print("Testing with custom local model...")
    print("Starting initialization...")
    print("-" * 70)

    start_time = time.time()
    
    # Import checks to give helpful errors if langchain-huggingface or sentence-transformers is missing
    try:
        import langchain_huggingface
        import sentence_transformers
    except ImportError as e:
        print(f"❌ Error: Required dependency missing. Please run 'uv pip install langchain-huggingface sentence-transformers' or 'pip install langchain-huggingface sentence-transformers'. Details: {e}")
        sys.exit(1)

    model = _try_hf_local_embeddings(model_id)
    
    if model is not None:
        elapsed = time.time() - start_time
        print("-" * 70)
        print(f"✅ Successfully initialized {model_id} in {elapsed:.2f} seconds!")
        
        # Test Query
        test_query = "What is Apple's total revenue for FY2024?"
        print(f"\nEmbedding single query: '{test_query}'")
        embed_start = time.time()
        test_vec = model.embed_query(test_query)
        embed_elapsed = time.time() - embed_start
        print(f"✅ Query embed successful (took {embed_elapsed:.3f}s)")
        print(f"✅ Dimensions: {len(test_vec)}")
        print(f"✅ Sample vector (first 5 values): {test_vec[:5]}")
        
        # Test Documents
        test_docs = [
            "Apple revenue was 394.3 billion for fiscal year 2024.",
            "Net income was 93.7 billion, a decrease of 3 percent."
        ]
        print(f"\nEmbedding batch of {len(test_docs)} documents...")
        batch_start = time.time()
        doc_vecs = model.embed_documents(test_docs)
        batch_elapsed = time.time() - batch_start
        print(f"✅ Batch embed successful (took {batch_elapsed:.3f}s)")
        print(f"✅ Number of vectors returned: {len(doc_vecs)}")
        print(f"✅ Vector dimensions: {len(doc_vecs[0]) if doc_vecs else 0}")
        print("=" * 70)
        print("  🎉 Local HuggingFace embeddings are fully functional on your system!")
        print("=" * 70)
    else:
        print("-" * 70)
        print(f"❌ Failed to initialize {model_id}.")
        print("Please check the log messages above for details (such as Out of Memory, package issues, or network errors).")
        print("Tip: If you want to run a quick test with a smaller model, pass it as a command line argument:")
        print("     python test/hf_local_test.py sentence-transformers/all-MiniLM-L6-v2")
        print("=" * 70)