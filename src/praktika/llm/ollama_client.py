"""Ollama backend over its native ``/api/chat`` endpoint.

The client posts ``format=<flat JSON schema>`` so the reply is grammar-constrained, sets
``num_ctx`` explicitly (Ollama's default of 2048 silently truncates transcripts), and disables
streaming. All traffic goes through an ``httpx.Client``; the default one is built with the egress
allow-list transport, so the client cannot reach a host outside ``allowed_hosts`` even when
misconfigured (C-01).
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import httpx

from praktika.config import AllowListTransport
from praktika.errors import LLMError, LLMSchemaError
from praktika.llm.base import parse_json_object
from praktika.logging import get_logger

log = get_logger(__name__)

DEFAULT_ALLOWED_HOSTS = ["localhost", "127.0.0.1"]
KEEP_ALIVE = "10m"


class OllamaClient:
    """``LLMClient`` for a local Ollama server.

    ``client`` is normally ``Settings.http_client()``; when omitted a client restricted to
    ``allowed_hosts`` (loopback by default) is created. ``timeout_s`` applies to the default
    client only.
    """

    name = "ollama"

    def __init__(
        self,
        base_url: str,
        model: str,
        num_ctx: int = 32768,
        timeout_s: int = 600,
        client: httpx.Client | None = None,
        *,
        allowed_hosts: list[str] | None = None,
    ) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.model = model
        self.num_ctx = int(num_ctx)
        self.timeout_s = int(timeout_s)
        self._client = client or httpx.Client(
            transport=AllowListTransport(allowed_hosts or DEFAULT_ALLOWED_HOSTS),
            timeout=float(timeout_s),
        )

    def with_model(self, model: str) -> OllamaClient:
        """Return a client for another model on the same server (used to degrade gracefully)."""
        return OllamaClient(self.base_url, model, self.num_ctx, self.timeout_s, self._client)

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        started = time.monotonic()
        try:
            response = self._client.post(url, json=body)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status, text = exc.response.status_code, exc.response.text[:300]
            raise LLMError(f"ollama {path} returned HTTP {status}: {text}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"ollama {path} request failed: {exc}") from exc
        try:
            payload = response.json()
        except ValueError as exc:
            raise LLMError(f"ollama {path} returned a non-JSON body") from exc
        log.debug("ollama.request", path=path, elapsed_ms=int((time.monotonic() - started) * 1000))
        if not isinstance(payload, dict):
            raise LLMError(f"ollama {path} returned an unexpected body type")
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
        """POST ``/api/chat`` with ``format=schema`` and return the parsed JSON object."""
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "format": schema,
            "options": {
                "temperature": temperature,
                "seed": seed,
                "num_ctx": self.num_ctx,
                "num_predict": max_tokens,
            },
            "stream": False,
            "keep_alive": KEEP_ALIVE,
        }
        payload = self._post("/api/chat", body)
        message = payload.get("message") or {}
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise LLMError("ollama /api/chat reply has no message.content")
        self._check_limits(payload, max_tokens)
        return parse_json_object(content)

    def _check_limits(self, payload: dict[str, Any], max_tokens: int) -> None:
        """Surface truncation instead of a confusing schema error later: a reply cut at
        ``num_predict`` (``done_reason == "length"``) raises ``LLMSchemaError``; a prompt that
        Ollama silently truncated at ``num_ctx`` (``prompt_eval_count`` at the window) is
        logged as a warning (the default 2048 window truncates silently)."""
        if payload.get("done_reason") == "length":
            raise LLMSchemaError(
                f"ollama reply truncated at num_predict={max_tokens}; raise max_tokens"
            )
        evaluated = payload.get("prompt_eval_count")
        if isinstance(evaluated, int) and evaluated >= self.num_ctx:
            log.warning(
                "ollama.prompt_truncated", prompt_eval_count=evaluated, num_ctx=self.num_ctx
            )

    def model_digest(self) -> str:
        """Digest of the served model from ``/api/show``.

        Ollama versions differ in whether ``/api/show`` carries a ``digest``; when it does not,
        a SHA-256 over the reported ``details`` and ``modelfile`` is used so provenance still
        changes whenever the model changes. Returns ``"unknown"`` if the server cannot answer.
        """
        try:
            payload = self._post("/api/show", {"model": self.model})
        except LLMError as exc:
            log.warning("ollama.show_failed", model=self.model, error=str(exc))
            return "unknown"
        digest = payload.get("digest")
        if isinstance(digest, str) and digest:
            return digest
        basis = json.dumps(
            {"details": payload.get("details"), "modelfile": payload.get("modelfile")},
            sort_keys=True,
            ensure_ascii=False,
        )
        return "sha256:" + hashlib.sha256(basis.encode("utf-8")).hexdigest()
