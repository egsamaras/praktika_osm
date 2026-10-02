"""Tests for the LLM clients and schema helpers; all HTTP is mocked."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from helpers_foundation import contains_key

from praktika.errors import EgressError, LLMError, LLMSchemaError
from praktika.llm.base import complete_model, flat_schema, parse_json_object
from praktika.llm.ollama_client import OllamaClient
from praktika.llm.openai_compat_client import OpenAICompatClient
from praktika.models import ChunkFindings, RetractionVerdict

VALID_VERDICT = {"retracted": False, "refs": [], "note": "No reversal found."}


class _Recorder:
    """Collects request bodies for a ``MockTransport`` and serves canned replies in order."""

    def __init__(self, replies: list[httpx.Response]) -> None:
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict[str, Any]] = []
        self._replies = list(replies)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.content:
            self.bodies.append(json.loads(request.content))
        return self._replies.pop(0) if self._replies else httpx.Response(500, text="exhausted")


def _client(rec: _Recorder) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(rec))


def _ollama_reply(content: str) -> httpx.Response:
    return httpx.Response(200, json={"message": {"role": "assistant", "content": content}})


def test_ollama_passes_format_and_num_ctx() -> None:
    rec = _Recorder([_ollama_reply(json.dumps(VALID_VERDICT))])
    client = OllamaClient(
        "http://127.0.0.1:11434/", "qwen2.5:14b", num_ctx=32768, client=_client(rec)
    )
    verdict = complete_model(
        client, "sys", "user", RetractionVerdict, temperature=0.2, seed=11, max_tokens=512
    )
    assert verdict.retracted is False
    assert rec.requests[0].url == "http://127.0.0.1:11434/api/chat"
    body = rec.bodies[0]
    assert body["model"] == "qwen2.5:14b"
    assert body["format"] == flat_schema(RetractionVerdict)
    assert body["options"] == {
        "temperature": 0.2,
        "seed": 11,
        "num_ctx": 32768,
        "num_predict": 512,
    }
    assert body["stream"] is False
    assert body["keep_alive"] == "10m"
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["messages"][0]["content"] == "sys"
    assert body["messages"][1]["content"] == "user"


def test_ollama_non_json_retry_then_schema_error() -> None:
    rec = _Recorder([_ollama_reply("I cannot answer that."), _ollama_reply('{"oops": 1}')])
    client = OllamaClient("http://127.0.0.1:11434", "qwen2.5:14b", client=_client(rec))
    with pytest.raises(LLMSchemaError):
        complete_model(client, "sys", "user", RetractionVerdict)
    assert len(rec.bodies) == 2, "exactly one retry"
    retry_user = rec.bodies[1]["messages"][1]["content"]
    assert retry_user.startswith("user")
    assert "did not match the schema" in retry_user
    assert "not JSON" in retry_user


def test_ollama_validation_error_retry_succeeds() -> None:
    bad = json.dumps({"retracted": "maybe", "refs": [], "note": "x"})
    rec = _Recorder([_ollama_reply(bad), _ollama_reply(json.dumps(VALID_VERDICT))])
    client = OllamaClient("http://127.0.0.1:11434", "qwen2.5:14b", client=_client(rec))
    assert complete_model(client, "s", "u", RetractionVerdict).note == "No reversal found."
    assert len(rec.bodies) == 2
    assert "retracted" in rec.bodies[1]["messages"][1]["content"]


def test_ollama_http_500_llm_error() -> None:
    rec = _Recorder([httpx.Response(500, text="boom")])
    client = OllamaClient("http://127.0.0.1:11434", "qwen2.5:14b", client=_client(rec))
    with pytest.raises(LLMError) as info:
        client.complete_json("s", "u", flat_schema(RetractionVerdict))
    assert not isinstance(info.value, LLMSchemaError)
    assert "500" in str(info.value)
    assert len(rec.requests) == 1, "transport errors are not retried"


def test_ollama_model_digest_from_show_and_fallback() -> None:
    rec = _Recorder(
        [
            httpx.Response(200, json={"digest": "sha256:abc"}),
            httpx.Response(200, json={"details": {"family": "qwen2"}, "modelfile": "FROM x"}),
            httpx.Response(404, text="missing"),
        ]
    )
    client = OllamaClient("http://127.0.0.1:11434", "qwen2.5:14b", client=_client(rec))
    assert client.model_digest() == "sha256:abc"
    assert rec.bodies[0] == {"model": "qwen2.5:14b"}
    assert rec.requests[0].url.path == "/api/show"
    assert client.model_digest().startswith("sha256:")
    assert client.model_digest() == "unknown"


def test_default_client_refuses_non_loopback_host() -> None:
    client = OllamaClient("http://api.example.com:11434", "qwen2.5:14b")
    with pytest.raises(EgressError):
        client.complete_json("s", "u", flat_schema(RetractionVerdict))


def test_openai_compat_response_format_strict() -> None:
    reply = {"choices": [{"message": {"role": "assistant", "content": json.dumps(VALID_VERDICT)}}]}
    rec = _Recorder([httpx.Response(200, json=reply)])
    client = OpenAICompatClient(
        "http://127.0.0.1:8000", "Qwen/Qwen3-32B", lambda: "tok-123", _client(rec)
    )
    out = complete_model(client, "sys", "user", RetractionVerdict, max_tokens=256)
    assert out.retracted is False
    req, body = rec.requests[0], rec.bodies[0]
    assert req.url == "http://127.0.0.1:8000/v1/chat/completions"
    assert req.headers["Authorization"] == "Bearer tok-123"
    rf = body["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["name"] == "RetractionVerdict"
    assert rf["json_schema"]["schema"] == flat_schema(RetractionVerdict)
    assert not contains_key(rf["json_schema"]["schema"], "$ref")
    assert body["max_tokens"] == 256 and body["stream"] is False
    assert body["model"] == "Qwen/Qwen3-32B"


def test_openai_compat_errors_and_digest() -> None:
    rec = _Recorder(
        [
            httpx.Response(503, text="overloaded"),
            httpx.Response(200, json={"data": [{"id": "Qwen/Qwen3-32B", "root": "/models/q3"}]}),
            httpx.Response(200, json={"data": []}),
        ]
    )
    client = OpenAICompatClient("http://127.0.0.1:8000", "Qwen/Qwen3-32B", None, _client(rec))
    with pytest.raises(LLMError):
        client.complete_json("s", "u", flat_schema(RetractionVerdict))
    assert "Authorization" not in rec.requests[0].headers
    assert client.model_digest() == "/models/q3"
    assert client.model_digest() == "unknown"


def test_flat_schema_has_no_refs() -> None:
    schema = flat_schema(ChunkFindings)
    assert schema["title"] == "ChunkFindings"
    assert not contains_key(schema, "$ref")
    assert "$defs" not in schema
    decision = schema["properties"]["decisions"]["items"]
    assert decision["properties"]["statement"]["maxLength"] == 400
    assert set(decision["properties"]["kind"]["enum"]) >= {"approved", "deferred"}
    # The original model schema is untouched.
    assert "$defs" in ChunkFindings.model_json_schema()


def test_parse_json_object_rejects_non_objects() -> None:
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(LLMSchemaError):
        parse_json_object("[1, 2]")
    with pytest.raises(LLMSchemaError):
        parse_json_object("plain prose")


def test_ollama_reports_truncation() -> None:
    """A reply cut at num_predict names the cause; a prompt at the context window warns."""
    from praktika.errors import LLMSchemaError

    cut = httpx.Response(
        200,
        json={
            "message": {"role": "assistant", "content": '{"retracted": fal'},
            "done_reason": "length",
        },
    )
    client = OllamaClient(
        "http://127.0.0.1:11434", "m", num_ctx=2048, client=_client(_Recorder([cut]))
    )
    with pytest.raises(LLMSchemaError, match="num_predict=64"):
        client.complete_json("s", "u", flat_schema(RetractionVerdict), max_tokens=64)
    full = httpx.Response(
        200,
        json={
            "message": {"role": "assistant", "content": json.dumps(VALID_VERDICT)},
            "done_reason": "stop",
            "prompt_eval_count": 2048,
        },
    )
    client = OllamaClient(
        "http://127.0.0.1:11434", "m", num_ctx=2048, client=_client(_Recorder([full]))
    )
    assert client.complete_json("s", "u", flat_schema(RetractionVerdict)) == VALID_VERDICT
