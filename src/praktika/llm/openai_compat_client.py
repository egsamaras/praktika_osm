"""OpenAI-compatible backend for vLLM behind your API gateway.

Uses ``/v1/chat/completions`` with ``response_format={"type": "json_schema", ...,
"strict": true}`` so vLLM's guided decoding (xgrammar) enforces the schema. The bearer token is
obtained from ``token_provider`` at every call and never stored on the instance.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import httpx

from praktika.config import AllowListTransport
from praktika.errors import LLMError
from praktika.llm.base import parse_json_object
from praktika.logging import get_logger

log = get_logger(__name__)

DEFAULT_ALLOWED_HOSTS = ["localhost", "127.0.0.1"]


class OpenAICompatClient:
    """``LLMClient`` for any OpenAI-compatible chat server (vLLM in the service profile).

    ``token_provider`` returns the current bearer token (``None`` for no auth). ``client`` is
    normally ``Settings.http_client()``; the default is restricted to loopback hosts.
    """

    name = "openai_compat"

    def __init__(
        self,
        base_url: str,
        model: str,
        token_provider: Callable[[], str | None] | None = None,
        client: httpx.Client | None = None,
        *,
        timeout_s: int = 600,
        allowed_hosts: list[str] | None = None,
    ) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.model = model
        self.timeout_s = int(timeout_s)
        self._token_provider = token_provider
        self._client = client or httpx.Client(
            transport=AllowListTransport(allowed_hosts or DEFAULT_ALLOWED_HOSTS),
            timeout=float(timeout_s),
        )

    def with_model(self, model: str) -> OpenAICompatClient:
        """Return a client for another model on the same server."""
        return OpenAICompatClient(
            self.base_url, model, self._token_provider, self._client, timeout_s=self.timeout_s
        )

    def _headers(self) -> dict[str, str]:
        token = self._token_provider() if self._token_provider else None
        return {"Authorization": f"Bearer {token}"} if token else {}

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        started = time.monotonic()
        try:
            response = self._client.request(method, url, json=body, headers=self._headers())
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            text = exc.response.text[:300]
            raise LLMError(f"{path} returned HTTP {exc.response.status_code}: {text}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"{path} request failed: {exc}") from exc
        try:
            payload = response.json()
        except ValueError as exc:
            raise LLMError(f"{path} returned a non-JSON body") from exc
        log.debug("openai_compat.request", path=path, ms=int((time.monotonic() - started) * 1000))
        return payload

    def complete_json(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        *,
        temperature: float = 0.0,
        seed: int = 7,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """POST ``/v1/chat/completions`` with a strict ``json_schema`` response format."""
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": str(schema.get("title", "Output")),
                    "schema": schema,
                    "strict": True,
                },
            },
            "temperature": temperature,
            "seed": seed,
            "max_tokens": max_tokens,
            "stream": False,
        }
        payload = self._request("POST", "/v1/chat/completions", body)
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("chat completion reply has no choices[0].message.content") from exc
        if not isinstance(content, str):
            raise LLMError("chat completion content is not a string")
        return parse_json_object(content)

    def model_digest(self) -> str:
        """The served model's ``root`` (vLLM reports the weights path/revision) or its id.

        Returns ``"unknown"`` if ``/v1/models`` cannot be read or does not list ``model``.
        """
        try:
            payload = self._request("GET", "/v1/models")
        except LLMError as exc:
            log.warning("openai_compat.models_failed", error=str(exc))
            return "unknown"
        data = payload.get("data", []) if isinstance(payload, dict) else []
        for entry in data:
            if isinstance(entry, dict) and entry.get("id") == self.model:
                return str(entry.get("root") or entry["id"])
        return "unknown"
