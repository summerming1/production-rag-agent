from __future__ import annotations

import os
from typing import Protocol


class Generator(Protocol):
    def generate(self, system: str, user: str) -> str: ...


class ExtractiveFallbackGenerator:
    """No-API fallback for demos/tests.

    This intentionally does *not* pretend to be an LLM. It exposes the first
    retrieved evidence block so the end-to-end retrieval/citation pipeline can
    be tested offline.
    """

    def generate(self, system: str, user: str) -> str:
        marker = "CONTEXT:\n"
        context = user.split(marker, 1)[1] if marker in user else user
        first = context.strip().split("\n\n", 1)[0]
        return "Evidence retrieved. " + first[:700]


class OpenAICompatibleGenerator:
    """Small adapter for vLLM or another OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install with: pip install -e '.[llm]'") from exc
        self.httpx = httpx
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.timeout_s = timeout_s

    def generate(self, system: str, user: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
        }
        with self.httpx.Client(timeout=self.timeout_s) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            body = response.json()
        return str(body["choices"][0]["message"]["content"])
