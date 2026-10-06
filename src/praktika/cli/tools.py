"""Utility commands: ``consent-script``, ``audio devices|check``, ``audit verify|tail``,
``config show`` and ``eval``.

``eval`` replays each golden meeting's canned LLM outputs (``PlaybackLLM``) through the real
pipeline and verifier, scores it with ``eval.checks`` and exits non-zero when a rollout gate
fails; ``--llm real`` uses the configured client instead.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import soundfile as sf
import typer

from praktika import consent
from praktika.audit import chain_report
from praktika.cli import context as ctx
from praktika.config import hide_credentials
from praktika.errors import LLMError, PraktikaError
from praktika.eval import checks
from praktika.eval import golden as golden_mod
from praktika.llm import pipeline
from praktika.llm import prompts as pr
from praktika.logging import get_logger

log = get_logger(__name__)

audio_app = typer.Typer(help="Audio devices and file checks.")
audit_app = typer.Typer(help="Hash-chained audit log.")
config_app = typer.Typer(help="Effective configuration.")

# Whole-word secret-like names only: ``llm_long_transcript_tokens`` is a size, not a secret.
SECRET_RE = re.compile(r"(?:^|_)(?:token|secret|password|api_key|key)$", re.IGNORECASE)
SILENCE_DB = -50.0


@ctx.guarded
def consent_script(
    lang: Annotated[str | None, typer.Option("--lang", help="en | ar (default: both)")] = None,
) -> None:
    """Print the spoken consent script and the chat banner (both languages by default)."""
    langs = ("en", "ar") if lang is None else (lang,)
    for code in langs:
        if code not in ("en", "ar"):
            raise PraktikaError("--lang must be en or ar")
        consent.print_script(ctx.console, code)  # type: ignore[arg-type]
        ctx.console.print("")


@audio_app.command("devices")
@ctx.guarded
def audio_devices() -> None:
    """List the input devices sounddevice can see."""
    try:
        import sounddevice as sd

        ctx.console.print(str(sd.query_devices()))
    except Exception as exc:  # sounddevice raises its own error types on hosts without audio
        raise PraktikaError(f"cannot list audio devices: {exc}") from exc


@audio_app.command("check")
@ctx.guarded
def audio_check(path: Annotated[Path, typer.Argument(help="WAV/FLAC/OGG file.")]) -> None:
    """Report duration, RMS level and silence ratio of an audio file."""
    try:
        data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    except (OSError, sf.LibsndfileError) as exc:
        raise PraktikaError(f"cannot read {path}: {exc}") from exc
    mono = data.mean(axis=1)
    duration = len(mono) / float(rate)
    rms = float(np.sqrt(np.mean(mono**2))) if len(mono) else 0.0
    rms_db = 20 * np.log10(rms) if rms > 0 else float("-inf")
    frame = max(1, int(rate * 0.02))
    frames = [mono[i : i + frame] for i in range(0, len(mono) - frame + 1, frame)]
    quiet = sum(1 for f in frames if (np.sqrt(np.mean(f**2)) or 1e-12) < 10 ** (SILENCE_DB / 20))
    ratio = quiet / len(frames) if frames else 1.0
    ctx.console.print(
        f"{path.name}: {duration:.1f} s, {rate} Hz, RMS {rms_db:.1f} dBFS, "
        f"silence {ratio:.0%}" + ("  (mostly silent)" if ratio >= 0.95 else "")
    )


@audit_app.command("verify")
@ctx.guarded
def audit_verify() -> None:
    """Verify ``audit.jsonl``: hash chain, ``audit.jsonl.head`` and the store's audit table must
    agree; exit 1 on a broken, truncated or missing chain."""
    rt = ctx.open_runtime(system_actor=True)
    path = Path(rt.settings.data_dir) / ctx.AUDIT_FILE
    try:
        report = chain_report(path, rt.store)
    finally:
        rt.close()
    if report.status == "ok":
        ctx.console.print(f"Audit chain valid: {report.detail}")
        return
    if report.status == "warn":
        ctx.console.print(f"Audit chain: {report.detail}")
        return
    raise PraktikaError(f"audit chain invalid: {report.detail}")


@audit_app.command("tail")
@ctx.guarded
def audit_tail(
    n: Annotated[int, typer.Option("-n", "--lines", help="Number of events.")] = 20,
) -> None:
    """Show the last N audit events (time, actor, event, meeting)."""
    settings = ctx.load_settings()
    path = Path(settings.data_dir) / ctx.AUDIT_FILE
    if not path.exists():
        ctx.console.print("No audit events yet.")
        return
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    for line in lines[-max(1, n) :]:
        e = json.loads(line)
        ctx.console.print(
            f"{e.get('ts')}  {e.get('actor')}  {e.get('event')}  {e.get('meeting_id') or ''}"
        )


@config_app.command("show")
@ctx.guarded
def config_show() -> None:
    """Print the effective settings as JSON (anything secret-like is masked, and so is the user
    name and password of any URL)."""
    settings = ctx.load_settings()
    data = settings.model_dump(mode="json")
    masked = {k: "***" if SECRET_RE.search(k) and v else _scrub(v) for k, v in data.items()}
    ctx.console.print(json.dumps(masked, indent=2, ensure_ascii=False))


def _scrub(value: Any) -> Any:
    """``value`` with ``hide_credentials`` applied to every string in it, lists and maps
    included."""
    if isinstance(value, str):
        return hide_credentials(value)
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items()}
    return value


class PlaybackLLM:
    """``LLMClient`` replaying a golden meeting's canned outputs keyed by schema title."""

    name = "playback"

    def __init__(self, playback: dict[str, list[dict[str, Any]]]) -> None:
        self.queues = {k: list(v) for k, v in playback.items()}

    def complete_json(
        self, system: str, user: str, schema: dict[str, Any], **kw: Any
    ) -> dict[str, Any]:
        title = str(schema.get("title", ""))
        queue = self.queues.get(title)
        if queue:
            return json.loads(json.dumps(queue.pop(0)))
        if title == "RetractionVerdict":
            return {"retracted": False, "refs": [], "note": "no playback entry"}
        raise LLMError(f"no playback output for schema {title!r}")

    def model_digest(self) -> str:
        return "sha256:playback"


class _NullAudit:
    def append(self, event: str, meeting_id: str | None = None, **detail: Any) -> None:
        return None


@ctx.guarded
def eval_cmd(
    golden: Annotated[Path, typer.Option("--golden", help="Golden set directory.")] = (
        golden_mod.GOLDEN_DIR
    ),
    llm: Annotated[str, typer.Option("--llm", help="fake | real")] = "fake",
    out: Annotated[Path, typer.Option("--out", help="Report path.")] = Path("eval_report.md"),
) -> None:
    """Run the golden set through the pipeline, write ``eval_report.md``, apply the gates."""
    if llm not in ("fake", "real"):
        raise PraktikaError("--llm must be fake or real")
    settings = ctx.load_settings()
    scores: list[dict[str, Any]] = []
    for gm in golden_mod.load_all(golden):
        client = PlaybackLLM(gm.playback) if llm == "fake" else ctx.llm_client(settings)
        prompts = pr.load(settings.prompts_dir, settings.prompt_version, gm.meeting.meeting_type)
        opts = pipeline.GenerateOptions(
            template=gm.meeting.meeting_type,
            prompt_version=settings.prompt_version,
            full_context_max_tokens=settings.llm_full_context_max_tokens,
        )
        try:
            minutes = pipeline.generate(
                gm.transcript, gm.meeting, client, prompts, "eval", None, opts, _NullAudit()
            )
        except PraktikaError as exc:
            log.warning("eval.meeting_failed", name=gm.name, error=str(exc))
            scores.append(checks.failed_score(gm.name, gm.language))
            continue
        scores.append(
            checks.score(
                minutes,
                gm.gold,
                gm.transcript,
                roster=gm.meeting.roster,
                name=gm.name,
                language=gm.language,
            )
        )
    report = checks.aggregate(scores)
    checks.write_report(report, out)
    passed, reasons = checks.gate(report)
    ctx.console.print(f"Evaluated {len(scores)} meeting(s); report written to {out}")
    for key, value in sorted(report.aggregate.items()):
        ctx.console.print(f"  {key:<28} {value if value is not None else 'n/a'}")
    if not passed:
        raise PraktikaError("evaluation gate failed: " + "; ".join(reasons))
    ctx.console.print("Gate: PASS")
