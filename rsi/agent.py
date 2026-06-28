"""
Agent runner for DO Model Studio.

Loads a provider (providers/do.json) and profile (profiles/*.json),
calls the chat completions endpoint, and returns the text response.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def build_client(provider_path: Path) -> OpenAI:
    provider = load_json(provider_path)
    api_key = os.environ.get(provider["api_key_env"], "")
    if not api_key:
        raise SystemExit(f"Missing env var: {provider['api_key_env']}")
    return OpenAI(api_key=api_key, base_url=provider["base_url"])


def run_agent(
    profile_path: Path,
    provider_path: Path,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Call the model and return the assistant's text response."""
    profile = load_json(profile_path)
    client = build_client(provider_path)

    response = client.chat.completions.create(
        model=profile["model"],
        max_tokens=profile.get("max_tokens", 8192),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content or ""
