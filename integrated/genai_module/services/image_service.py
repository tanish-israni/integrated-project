import os
import base64
from urllib.parse import quote

import requests
import streamlit as st

from config.config import HF_API_URL, HF_MODEL


def _get_huggingface_api_key() -> str | None:
    return os.getenv("HUGGINGFACE_API_KEY") or st.secrets.get("HUGGINGFACE_API_KEY")


def generate_image(prompt: str) -> bytes:
    api_key = _get_huggingface_api_key()
    if not api_key:
        raise ValueError("Missing HUGGINGFACE_API_KEY.")

    headers = {"Authorization": f"Bearer {api_key}"}

    hf_error: str | None = None
    try:
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json={
                "model": HF_MODEL,
                "prompt": prompt,
                "size": "1024x1024",
            },
            timeout=180,
        )
        if response.status_code in (200, 201):
            payload = response.json()
            data = payload.get("data", [])
            if data:
                first = data[0]
                if "b64_json" in first:
                    return base64.b64decode(first["b64_json"])
                if "url" in first:
                    image_response = requests.get(first["url"], timeout=60)
                    image_response.raise_for_status()
                    return image_response.content
            hf_error = f"unexpected response: {response.text[:300]}"
        else:
            hf_error = response.text[:300]
    except requests.exceptions.RequestException as exc:
        hf_error = str(exc)

    # Fallback provider for reliability when HF model endpoints change/deprecate.
    try:
        fallback_url = f"https://image.pollinations.ai/prompt/{quote(prompt)}?width=1024&height=1024&nologo=true"
        fallback_response = requests.get(fallback_url, timeout=90)
        fallback_response.raise_for_status()
        return fallback_response.content
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"Hugging Face API error: {hf_error}. Fallback image provider error: {exc}"
        ) from exc
