import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from huggingface_hub import InferenceClient
from config import settings

client = InferenceClient(
    provider="hf-inference",
    api_key=settings.hf_token
)

result = client.feature_extraction(
    "Today is a sunny day and I will get some ice cream.",
    model="Qwen/Qwen3-Embedding-0.6B",
)