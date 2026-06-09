"""Shared LLM factory for all agents.

Uses ckey as an OpenAI-compatible API.
"""

import os

from langchain_openai import ChatOpenAI


def get_llm() -> ChatOpenAI:
    """Return a ChatOpenAI client pointed at ckey."""
    api_key = os.getenv("ckey_api_key", "").strip()
    if not api_key or api_key in {"your_key_here", "your_openrouter_key_here"}:
        raise RuntimeError(
            "ckey_api_key is required. Copy .env.example to .env and set a real ckey API key."
        )
    return ChatOpenAI(
        model=os.getenv("ckey_model", "GPT-5.4-mini"),
        temperature=0.3,
        openai_api_key=api_key,
        openai_api_base=os.getenv("Base_url", "https://api.xah.io/v1"),
    )
