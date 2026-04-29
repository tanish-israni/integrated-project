import os

import requests
import streamlit as st

from config.config import HF_API_URL


def _get_huggingface_api_key() -> str | None:
    return os.getenv("HUGGINGFACE_API_KEY") or st.secrets.get("HUGGINGFACE_API_KEY")


def generate_image(prompt: str) -> bytes:
    api_key = _get_huggingface_api_key()
    if not api_key:
        raise ValueError("Missing HUGGINGFACE_API_KEY.")

    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json={"inputs": prompt},
            timeout=120,
        )
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Hugging Face API error: {exc}") from exc

    if response.status_code != 200:
        try:
            details = response.json()
        except ValueError:
            details = response.text
        raise RuntimeError(f"Hugging Face API error: {details}")

    return response.content
