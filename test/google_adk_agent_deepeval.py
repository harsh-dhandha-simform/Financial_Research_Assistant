import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai import types
from deepeval.integrations.google_adk import instrument_google_adk
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.evaluate.configs import AsyncConfig
from deepeval.metrics import TaskCompletionMetric
from deepeval.models import GeminiModel
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

instrument_google_adk()

agent = LlmAgent(model="gemini-2.0-flash", name="assistant", instruction="Be concise.")
runner = InMemoryRunner(agent=agent, app_name="deepeval-google-adk")

async def run_agent(prompt: str) -> str:
    session = await runner.session_service.create_session(app_name="deepeval-google-adk", user_id="demo-user")
    message = types.Content(role="user", parts=[types.Part(text=prompt)])
    async for event in runner.run_async(user_id="demo-user", session_id=session.id, new_message=message):
        if event.is_final_response() and event.content:
            return "".join(part.text for part in event.content.parts if getattr(part, "text", None))
    return ""

# Goldens are the inputs you want to evaluate.
dataset = EvaluationDataset(goldens=[Golden(input="What is 7 multiplied by 8?")])

# We use the same Gemini model for evaluation
eval_model = GeminiModel(model="gemini-2.0-flash")

# `evals_iterator` loops through goldens and applies metrics.
for golden in dataset.evals_iterator(async_config=AsyncConfig(run_async=True), metrics=[TaskCompletionMetric(model=eval_model)]):
    task = asyncio.create_task(run_agent(golden.input)) # Produces trace for evaluation
    dataset.evaluate(task)