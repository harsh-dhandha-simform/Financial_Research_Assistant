"""
Pydantic model for Human-in-the-Loop (HITL) feedback.

At the interrupt_before checkpoint (Module 15), the graph pauses
and presents the Synthesis Agent's output for human review.
The reviewer submits a HumanFeedback to approve, reject, or modify
the analysis before it proceeds to the output fork.
"""

from enum import Enum

from pydantic import BaseModel, Field

from schemas.agents import InvestmentRating


class FeedbackAction(str, Enum):
    """Actions the human reviewer can take at the HITL checkpoint."""

    APPROVE = "approve"
    MODIFY = "modify"
    REGENERATE = "regenerate"
    REJECT = "reject"


class HumanFeedback(BaseModel):
    """Human reviewer's feedback at the HITL interrupt checkpoint.

    When action is APPROVE, the graph continues to the output fork as-is.
    When action is MODIFY, the graph uses the overrides below.
    When action is REGENERATE, the graph re-runs the agent pipeline.
    When action is REJECT, the graph terminates without producing reports.
    """

    action: FeedbackAction = Field(..., description="Review decision")
    reviewer_notes: str = Field(
        default="",
        description="Free-text notes from the reviewer",
    )

    # ── Optional overrides (used when action is MODIFY) ──────────────────────
    override_rating: InvestmentRating | None = Field(
        default=None,
        description="Override the investment rating (MODIFY only)",
    )
    override_thesis: str = Field(
        default="",
        description="Override or amend the investment thesis text (MODIFY only)",
    )
    additional_risks: list[str] = Field(
        default_factory=list,
        description="Additional risks the reviewer wants included",
    )
    additional_strengths: list[str] = Field(
        default_factory=list,
        description="Additional strengths the reviewer wants included",
    )
