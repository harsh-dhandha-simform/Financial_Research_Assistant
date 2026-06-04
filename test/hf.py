import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from huggingface_hub import InferenceClient
from config import settings
client = InferenceClient(
    api_key=settings.hf_token
)


stream = client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct:groq",
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
    stream=True,
)

for chunk in stream:
    print(chunk.choices[0].delta.content, end="")