"""Optional: expand a user query into retrieval-friendly phrasing(s)."""
from __future__ import annotations

from .groq_client import GroqLLM

SYSTEM = (
    "You rewrite legal-contract search queries to make keyword and semantic "
    "retrieval more effective. You NEVER answer the question. "
    "Return 1 improved query on a single line. Keep legal terms verbatim."
)


def rewrite(query: str, llm: GroqLLM | None = None) -> str:
    llm = llm or GroqLLM()
    out = llm.chat(SYSTEM, query, max_tokens=120).strip().splitlines()
    return (out[0] if out else query).strip() or query
