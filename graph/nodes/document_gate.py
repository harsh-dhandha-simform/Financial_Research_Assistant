"""
Document Gate node — verifies that the pipeline has enough context to proceed.

Runs after the query guardrail and before the supervisor.
If no documents were ingested and the query requires them, it can reject
and route to END. Currently it acts as a pass-through unless specific
conditions are met.
"""

import logging
from typing import Any

from graph.state import ResearchState
from langfuse.decorators import observe

logger = logging.getLogger(__name__)

@observe()
def document_gate_node(state: ResearchState) -> dict[str, Any]:
    """Check if the pipeline has what it needs to proceed."""
    logger.info("Document Gate: checking readiness for %s", state.company_name)
    
    # If we need to enforce that documents MUST be ingested, we could do:
    # if not state.context and some_other_condition:
    #     return {"gate_rejected": True, "rejection_message": "No documents ingested."}
    
    # For now, we allow the pipeline to proceed even without documents
    # because the News Agent can use Tavily web search.
    return {
        "gate_rejected": False,
        "rejection_message": ""
    }
