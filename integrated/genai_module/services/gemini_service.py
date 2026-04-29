import os

import streamlit as st
from openai import OpenAI

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"


def _get_groq_api_key() -> str | None:
    return os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")


def generate_text(prompt: str, retrieved_context: list[dict] | None = None) -> str:
    api_key = _get_groq_api_key()
    if not api_key:
        raise ValueError("Missing GROQ_API_KEY.")

    client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
    model_name = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
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
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a hospitality concept writer. Provide short, vivid, practical concept text."
                    ),
                },
                {"role": "user", "content": request_text},
            ],
            temperature=0.7,
        )
    except Exception as exc:
        raise RuntimeError(f"Groq API error: {exc}") from exc

    text = ""
    if response.choices and response.choices[0].message:
        text = response.choices[0].message.content or ""
    if not text or not text.strip():
        raise RuntimeError("Groq returned an empty response.")

    return text.strip()
