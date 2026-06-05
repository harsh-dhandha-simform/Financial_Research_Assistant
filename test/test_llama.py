from openrouter import OpenRouter
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
with OpenRouter(
  api_key=os.getenv("OPENROUTER_API_KEY", ""),
) as client:
  response = client.chat.send(
    model="meta-llama/llama-3.3-70b-instruct:free",
    messages=[
      {
        "role": "user",
        "content": "What is the meaning of life?"
      }
    ]
  )

  print(response.choices[0].message.content)