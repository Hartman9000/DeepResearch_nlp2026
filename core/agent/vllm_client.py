import json
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


class VLLMClient:
    def __init__(self, base_url: str, api_key: str = "dummy") -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def chat_completions(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            request_summary = {
                "model": payload.get("model"),
                "messages": len(payload.get("messages", [])),
                "max_tokens": payload.get("max_tokens"),
                "has_tools": "tools" in payload,
                "tool_choice": payload.get("tool_choice"),
                "payload_bytes": len(data),
            }
            print(
                "vLLM HTTP error\n"
                f"status: {exc.code} {exc.reason}\n"
                f"url: {request.full_url}\n"
                f"request_summary: {json.dumps(request_summary, ensure_ascii=False)}\n"
                f"response_body:\n{body}"
            )
            raise

    def simple_chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if extra_payload:
            payload.update(extra_payload)
        return self.chat_completions(payload)
