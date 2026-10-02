"""``praktika models pull|verify|register``: the model register (C-13).

``pull`` mirrors a role from the Hub with ``HF_TOKEN`` read from this command's environment
only; ``register`` records weights that arrived by other means (air-gapped hosts); ``verify``
re-hashes everything against ``models.yaml`` and exits 1 on any mismatch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.measure import Measurement
from rich.table import Table

from praktika.cli import context as ctx
from praktika.errors import PraktikaError
from praktika.models_registry import manage

models_app = typer.Typer(help="Mirror, register and verify model weights.")

#: Upper bound for measuring a table's natural width (far wider than any real row).
MAX_TABLE_WIDTH = 1000

RoleArg = Annotated[str, typer.Argument(help="stt_en | stt_ar | stt_ar_full | diarize")]


@models_app.command("pull")
@ctx.guarded
def models_pull(role: RoleArg) -> None:
    """Mirror a model repo into ``models_dir/<role>`` and record its hashes."""
    settings = ctx.load_settings()
    try:
        record = manage.pull(role, settings)
    except ValueError as exc:
        raise PraktikaError(str(exc)) from exc
    except Exception as exc:  # Hub / conversion errors are third-party types
        raise PraktikaError(f"pull failed for {role}: {exc}") from exc
    ctx.console.print(
        f"Pulled {record.role} from {record.repo}@{record.revision} "
        f"({len(record.files_sha256)} files) into {record.local_path}"
    )


@models_app.command("register")
@ctx.guarded
def models_register(
    role: RoleArg,
    path: Annotated[Path, typer.Argument(help="Directory holding the weights.")],
    repo: Annotated[str | None, typer.Option("--repo", help="Source repo id.")] = None,
    revision: Annotated[str, typer.Option("--revision", help="Source revision.")] = "local",
    licence: Annotated[str | None, typer.Option("--licence", help="Licence name.")] = None,
) -> None:
    """Register already-mirrored weights (no download) and hash every file."""
    settings = ctx.load_settings()
    try:
        record = manage.register(
            role, path, settings, repo=repo, revision=revision, licence=licence
        )
    except FileNotFoundError as exc:
        raise PraktikaError(str(exc)) from exc
    ctx.console.print(
        f"Registered {record.role}: {len(record.files_sha256)} files at {record.local_path} "
        f"(register: {manage.register_path(settings)})"
    )


def print_full_width(table: Table) -> None:
    """Print ``table`` on the CLI console at the width its longest values need, never
    narrower than the console, so no cell is cut short: the repo and the full 40-character
    revision stay whole when the output is piped, teed or shown in an 80-column window (the
    install record keeps that line as evidence that the revision matches the one shipped)."""
    console = ctx.console
    natural = Measurement.get(console, console.options.update(width=MAX_TABLE_WIDTH), table)
    if natural.maximum <= console.width:
        console.print(table)
        return
    # ``Console.print(width=...)`` never exceeds the console's own width, so a console that
    # writes to the same stream at the table's natural width does the printing.
    wide = Console(
        file=console.file,
        width=natural.maximum,
        force_terminal=console.is_terminal,
        markup=False,
        highlight=False,
        emoji=False,
        soft_wrap=True,
    )
    wide.print(table)


@models_app.command("verify")
@ctx.guarded
def models_verify() -> None:
    """Re-hash every registered model; exit 1 on the first difference. The table prints the
    full repo and revision of each model, whatever the terminal width."""
    settings = ctx.load_settings()
    records = manage.verify(settings)
    table = Table("Role", "Repo", "Revision", "Files", "Licence")
    for r in records:
        table.add_row(r.role, r.repo, r.revision, str(len(r.files_sha256)), r.licence)
    print_full_width(table)
    ctx.console.print("All registered models match their recorded hashes.")
