"""
Failure log curator — the third small LLM role.

Reads a failed sol.py + results.json and produces a structured
training pair in train.jsonl format for the LoRA trainer.

Uses a small local model (via llama.cpp server) or falls back to
the DO Model Studio API when running without a local model.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


SYSTEM_PROMPT = """You are a training data curator for a self-improving AI system.
Given a failed Triton kernel solution and its error output, produce a single
JSON training pair that teaches the model how to fix the specific mistake.

Output ONLY a JSON object with two keys:
  "prompt": the instruction to fix the error (be specific about what went wrong)
  "completion": the corrected Triton kernel code

No explanation, no markdown fences around the JSON itself."""


def _extract_error(results: dict) -> str:
    parts = []
    if results.get("error_message"):
        parts.append(results["error_message"])
    if results.get("error_details"):
        parts.append(results["error_details"])
    for r in results.get("details", []):
        if r.get("error"):
            parts.append(r["error"])
    return "\n".join(parts) if parts else f"Status: {results.get('status', 'UNKNOWN')}"


def _extract_code_block(text: str) -> str:
    """Pull the first ```python ... ``` block, or return text as-is."""
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def _parse_pair(raw: str) -> dict | None:
    raw = raw.strip()
    # Strip outer markdown fences if the LLM wrapped the JSON
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        obj = json.loads(raw)
        if "prompt" in obj and "completion" in obj:
            return obj
    except json.JSONDecodeError:
        pass
    return None


def curate_via_api(
    failed_code: str,
    results: dict,
    profile_path: Path,
    provider_path: Path,
) -> dict | None:
    """Use DO Model Studio (small/fast model) to generate a training pair."""
    from rsi.agent import run_agent

    error_text = _extract_error(results)
    user_prompt = (
        f"Failed Triton kernel:\n```python\n{failed_code}\n```\n\n"
        f"Error output:\n{error_text}\n\n"
        "Produce the training pair JSON now."
    )
    raw = run_agent(profile_path, provider_path, SYSTEM_PROMPT, user_prompt)
    return _parse_pair(raw)


def curate_via_local_server(
    failed_code: str,
    results: dict,
    base_url: str,
    max_tokens: int = 2048,
) -> dict | None:
    """Use a running llama-server (OpenAI-compat) to generate a training pair."""
    from openai import OpenAI

    error_text = _extract_error(results)
    user_prompt = (
        f"Failed Triton kernel:\n```python\n{failed_code}\n```\n\n"
        f"Error output:\n{error_text}\n\n"
        "Produce the training pair JSON now."
    )
    try:
        client = OpenAI(api_key="no-key", base_url=base_url)
        response = client.chat.completions.create(
            model="local",
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = response.choices[0].message.content or ""
        return _parse_pair(raw)
    except Exception as e:
        print(f"  [curate] local server error: {e}")
        return None


def curate_synthetic(task_md: str, correct_code: str) -> dict:
    """
    Produce a training pair from a CORRECT solution (no LLM needed).
    Used in Gen 0 to seed train.jsonl before any failures occur.
    """
    return {
        "prompt": (
            "Write a correct Triton kernel that solves the following task:\n\n"
            + task_md.strip()
        ),
        "completion": correct_code.strip(),
    }


def append_to_jsonl(pair: dict, jsonl_path: Path) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(pair) + "\n")
    print(f"  [curate] appended training pair → {jsonl_path} ({_count_lines(jsonl_path)} total)")


def _count_lines(path: Path) -> int:
    try:
        return sum(1 for _ in open(path))
    except FileNotFoundError:
        return 0
