import json
import re
from typing import Any, Dict, List, Optional, Tuple


def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    return text.strip()


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON extraction for small local models that add prose."""
    if not text:
        return None

    visible = strip_thinking(text)
    candidates = [visible, text.strip()]

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", visible, flags=re.DOTALL)
    if fenced:
        candidates.insert(0, fenced.group(1))

    first = visible.find("{")
    last = visible.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidates.insert(0, visible[first : last + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def append_agent_exchange(
    messages: List[Dict[str, Any]],
    system_prompt: str,
    user_content: str,
    assistant_content: str,
) -> None:
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": assistant_content})


def call_json_agent(
    client: Any,
    model: str,
    system_prompt: str,
    user_content: str,
    max_tokens: int,
) -> Tuple[Dict[str, Any], str]:
    response = client.simple_chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    raw = response["choices"][0]["message"].get("content", "")
    parsed = extract_json_object(raw)
    if parsed is None:
        raise ValueError(f"Model did not return valid JSON:\n{raw}")
    return parsed, raw
