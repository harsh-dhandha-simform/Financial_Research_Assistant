"""
API module — FastAPI layer for the Financial Research Analyst.

Run: uvicorn api.main:app --port 8000 --reload
"""

from api.main import app

__all__ = ["app"]
