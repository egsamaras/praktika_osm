"""The ``praktika`` command-line application.

``app`` is the typer application registered as the ``praktika`` console script. Commands live
in sibling modules: ``meetings`` (start, ingest), ``meeting_ops`` (abort, transcribe,
generate), ``review_cmd``
(approve, export, serve, search, actions), ``ops`` (retention, hold, dsar), ``tools``
(consent-script, audio, audit, config, eval), ``doctor`` and ``models_cmd``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import typer
from typer.core import TyperGroup

from praktika.cli import context as ctx
from praktika.cli import doctor, meeting_ops, meetings, ops, review_cmd, tools
from praktika.cli.models_cmd import models_app
from praktika.logging import configure_logging

if TYPE_CHECKING:
    import click

HELP_OPTION = "--help"
HELP_REQUESTED = "praktika.help_requested"


def help_requested(args: list[str]) -> bool:
    """True when ``--help`` is among ``args`` before any ``--`` separator."""
    head = args[: args.index("--")] if "--" in args else args
    return HELP_OPTION in head


class PraktikaGroup(TyperGroup):
    """The root command group. It notes whether ``--help`` was asked for anywhere on the
    command line: the root callback runs before a subcommand parses its own ``--help``, and
    help must print even when the env file is missing or unreadable."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        ctx.meta[HELP_REQUESTED] = help_requested(list(args))
        return super().parse_args(ctx, args)


app = typer.Typer(
    name="praktika",
    cls=PraktikaGroup,
    help="On-premises English meeting minutes with evidence citations and human approval.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode=None,
    pretty_exceptions_enable=False,
)


@app.callback()
def _root(
    context: typer.Context,
    log_level: Annotated[str, typer.Option("--log-level", help="DEBUG|INFO|WARNING")] = "WARNING",
    json_logs: Annotated[bool, typer.Option("--json-logs", help="JSON log lines.")] = False,
) -> None:
    """Configure logging for every command, and refuse to run at all when the env file is
    missing or unreadable (``context.checked_env_file``). ``--help`` on any subcommand skips
    the env-file check, so the help text can be read before the env file is written."""
    configure_logging(json=json_logs, level=log_level)
    if context.resilient_parsing or context.meta.get(HELP_REQUESTED):
        return
    ctx.checked_env_file()


app.command("doctor")(doctor.doctor)
app.command("start")(meetings.start)
app.command("ingest")(meetings.ingest)
app.command("abort")(meeting_ops.abort)
app.command("transcribe")(meeting_ops.transcribe)
app.command("generate")(meeting_ops.generate)
app.command("serve")(review_cmd.serve)
app.command("approve")(review_cmd.approve)
app.command("export")(review_cmd.export)
app.command("search")(review_cmd.search)
app.command("actions")(review_cmd.actions)
app.command("consent-script")(tools.consent_script)
app.command("eval")(tools.eval_cmd)
app.add_typer(ops.retention_app, name="retention")
app.add_typer(ops.hold_app, name="hold")
app.add_typer(ops.dsar_app, name="dsar")
app.add_typer(models_app, name="models")
app.add_typer(tools.audio_app, name="audio")
app.add_typer(tools.audit_app, name="audit")
app.add_typer(tools.config_app, name="config")

__all__ = ["app"]
