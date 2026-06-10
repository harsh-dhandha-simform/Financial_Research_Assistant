"""
Memo renderer — formats and sends an InvestmentMemo to Chainlit.

Uses the existing markdown_exporter to convert the memo, then sends
it as a Chainlit message.
"""

import chainlit as cl

from output.markdown_exporter import memo_to_markdown
from schemas.reports import InvestmentMemo


async def render_memo(memo: InvestmentMemo):
    """Render an InvestmentMemo as a formatted Chainlit message."""
    memo_md = memo_to_markdown(memo)
    rating = memo.rating.value.replace("_", " ").upper()

    await cl.Message(
        content=(
            f"## 📋 Investment Memo\n\n"
            f"**Rating:** {rating}\n\n"
            f"---\n\n"
            f"{memo_md}"
        )
    ).send()
