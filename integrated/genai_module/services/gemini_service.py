import os

import streamlit as st
from google import genai

from config.config import GEMINI_MODEL


def _get_gemini_api_key() -> str | None:
    return os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")


def generate_text(prompt: str, retrieved_context: list[dict] | None = None) -> str:
    api_key = _get_gemini_api_key()
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY.")

    client = genai.Client(api_key=api_key)
    context_section = ""

    if retrieved_context:
        context_lines = []
        for index, item in enumerate(retrieved_context, start=1):
            context_lines.append(
                f"Reference {index} prompt: {item['prompt']}\n"
                f"Reference {index} concept: {item['generated_text']}"
            )
        context_section = (
            "Use the following previously saved user concepts as style and preference context. "
            "Stay consistent with them when relevant, but still answer the new prompt directly.\n\n"
            + "\n\n".join(context_lines)
        )

    request_text = (
        "Write a short, vivid hospitality concept description for the following prompt.\n\n"
        f"User prompt: {prompt}"
    )

    if context_section:
        request_text = f"{context_section}\n\n{request_text}"

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=request_text,
        )
    except Exception as exc:
        raise RuntimeError(f"Gemini API error: {exc}") from exc

    text = getattr(response, "text", "")
    if not text or not text.strip():
        raise RuntimeError("Gemini returned an empty response.")

    return text.strip()
