import os
import base64

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
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Hugging Face API error: {exc}") from exc

    if response.status_code not in (200, 201):
        try:
            details = response.json()
        except ValueError:
            details = response.text
        raise RuntimeError(f"Hugging Face API error: {details}")

    payload = response.json()
    data = payload.get("data", [])
    if not data:
        raise RuntimeError(f"Hugging Face API error: unexpected response: {payload}")

    first = data[0]
    if "b64_json" in first:
        return base64.b64decode(first["b64_json"])
    if "url" in first:
        image_response = requests.get(first["url"], timeout=60)
        image_response.raise_for_status()
        return image_response.content

    raise RuntimeError(f"Hugging Face API error: unsupported response format: {payload}")
