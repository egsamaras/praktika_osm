"""LLM client protocol, schema flattening and validated completion.

Every backend (Ollama, vLLM's OpenAI-compatible server, the test ``FakeLLM``) exposes one
method, ``complete_json``, that takes a system prompt, a user prompt and a flat JSON schema and
returns a parsed JSON object. ``complete_model`` wraps it with Pydantic validation and exactly one
retry; ``flat_schema`` produces the ``$ref``-free schema the constrained decoders need.
"""

from __future__ import annotations

import copy
import json
import time
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ValidationError

from praktika.errors import LLMSchemaError
from praktika.logging import get_logger

log = get_logger(__name__)


@runtime_checkable
class LLMClient(Protocol):
    """A JSON-schema constrained chat completion backend."""

    name: str

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
        """Return the model's JSON object for ``schema``.

        Raises ``LLMError`` on transport or HTTP failure and ``LLMSchemaError`` when the model's
        reply is not a JSON object at all.
        """
        ...

    def model_digest(self) -> str:
        """A stable identifier of the served weights (for provenance), or ``"unknown"``."""
        ...


def flat_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return ``model``'s JSON schema with every ``$defs`` entry inlined and no ``$ref`` left.

    Ollama's ``format`` and vLLM's ``json_schema`` grammars handle nested objects but not
    references. Domain models are flat and non-recursive by design; a self-referencing model
    raises ``RecursionError`` rather than looping.
    """
    schema = copy.deepcopy(model.model_json_schema())
    defs = schema.pop("$defs", {})

    def walk(node: Any, stack: tuple[str, ...]) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                name = str(node["$ref"]).rsplit("/", 1)[-1]
                if name in stack:
                    raise RecursionError(f"recursive schema via {name}")
                target = copy.deepcopy(defs[name])
                extra = {k: v for k, v in node.items() if k != "$ref"}
                return walk({**target, **extra}, (*stack, name))
            return {k: walk(v, stack) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v, stack) for v in node]
        return node

    return walk(schema, ())


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse ``text`` as a JSON object; raise ``LLMSchemaError`` if it is not one.

    Tolerates a Markdown code fence around the object, which some models add despite
    constrained decoding.
    """
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.startswith("json"):
            candidate = candidate[4:]
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise LLMSchemaError(f"model reply is not JSON: {exc.msg} at char {exc.pos}") from exc
    if not isinstance(obj, dict):
        raise LLMSchemaError(f"model reply is JSON but not an object: {type(obj).__name__}")
    return obj


def complete_model[T: BaseModel](
    client: LLMClient,
    system: str,
    user: str,
    model: type[T],
    *,
    temperature: float = 0.0,
    seed: int = 7,
    max_tokens: int = 4096,
) -> T:
    """Call ``client`` with ``flat_schema(model)`` and validate the reply as ``model``.

    On a validation failure (or a non-JSON reply) the call is retried exactly once with the
    error text appended to the user message; a second failure raises ``LLMSchemaError``.
    Transport errors (``LLMError``) propagate unchanged.
    """
    schema = flat_schema(model)
    kw = {"temperature": temperature, "seed": seed, "max_tokens": max_tokens}
    prompt = user
    last_error = ""
    for attempt in (1, 2):
        try:
            raw = client.complete_json(system, prompt, schema, **kw)
            return model.model_validate(raw)
        except (ValidationError, LLMSchemaError) as exc:
            last_error = str(exc)
            log.warning(
                "llm.schema_retry", schema=model.__name__, attempt=attempt, error=last_error[:500]
            )
            prompt = (
                f"{user}\n\nYour previous reply did not match the schema. Error:\n{last_error}\n"
                "Reply again with valid JSON matching the schema and nothing else."
            )
    raise LLMSchemaError(
        f"{model.__name__}: reply failed schema validation after one retry: {last_error}"
    )


def generator_name(client: Any) -> str:
    """``client.model`` when the backend exposes one, else its protocol ``name``."""
    return str(getattr(client, "model", None) or getattr(client, "name", "unknown"))


class AuditLike(Protocol):
    """The slice of ``audit.AuditLog`` the pipeline and the ingest path need.

    ``classification``, ``object``, ``model`` and ``prompt_sha`` populate the corresponding
    ``AuditEvent`` fields; everything else lands in ``detail``. A fake must accept them.
    """

    def append(
        self,
        event: str,
        meeting_id: str | None,
        *,
        classification: str | None = None,
        object: str | None = None,  # noqa: A002 - field name fixed by AuditEvent
        model: str | None = None,
        prompt_sha: str | None = None,
        **detail: Any,
    ) -> Any: ...


class AuditedClient:
    """Delegates to an ``LLMClient`` and emits one ``llm.call`` audit event per call.

    The event carries the schema name, model, prompt hash, timing and sizes only; never prompt
    or reply content. It is emitted even when the inner call raises. ``classification`` is the
    meeting's classification value (for example ``"internal"``); every caller that holds the
    meeting passes it, so a SIEM rule keyed on classification sees each model call.
    """

    def __init__(
        self,
        inner: LLMClient,
        audit: AuditLike,
        meeting_id: str,
        prompt_sha: str,
        *,
        classification: str | None = None,
    ):
        self.inner, self.audit = inner, audit
        self.meeting_id, self.prompt_sha = meeting_id, prompt_sha
        self.classification = classification
        self.name = inner.name

    def complete_json(
        self, system: str, user: str, schema: dict[str, Any], **kw: Any
    ) -> dict[str, Any]:
        started = time.monotonic()
        try:
            return self.inner.complete_json(system, user, schema, **kw)
        finally:
            self.audit.append(
                "llm.call",
                self.meeting_id,
                classification=self.classification,
                schema=str(schema.get("title", "")),
                model=generator_name(self.inner),
                prompt_sha=self.prompt_sha,
                elapsed_ms=int((time.monotonic() - started) * 1000),
                user_chars=len(user),
                **{k: kw[k] for k in ("temperature", "seed", "max_tokens") if k in kw},
            )

    def model_digest(self) -> str:
        return self.inner.model_digest()
