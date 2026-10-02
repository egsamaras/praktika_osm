"""Where exports may be written (C-01: nothing leaves the box by default).

``resolve_export_target`` returns the file to write: ``data_dir/exports/<name>`` (directory
created 0700) unless ``--out`` was given. A target inside iCloud Drive
(``~/Library/Mobile Documents``) is refused without ``force``; a target under ``~/Documents``
or ``~/Desktop`` while iCloud "Desktop & Documents" sync is on is refused too, because Apple
uploads those folders; any other ``~/Documents``/``~/Desktop`` target gets a warning.
"""

from __future__ import annotations

import os
from pathlib import Path

from praktika.config import Settings
from praktika.errors import PraktikaError
from praktika.logging import get_logger

log = get_logger(__name__)

EXPORTS_SUBDIR = "exports"
ICLOUD_ROOT = Path("~/Library/Mobile Documents").expanduser()
ICLOUD_DOCS = ICLOUD_ROOT / "com~apple~CloudDocs" / "Documents"
SYNCED_HOME_DIRS = ("Documents", "Desktop")


def _under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def icloud_desktop_documents_enabled() -> bool:
    """True when iCloud "Desktop & Documents" sync appears to be on for this account."""
    return ICLOUD_DOCS.is_dir()


def cloud_synced(path: Path) -> str | None:
    """A reason when ``path`` would be uploaded by iCloud, else ``None``."""
    resolved = Path(path).expanduser().resolve()
    if _under(resolved, ICLOUD_ROOT.resolve()) if ICLOUD_ROOT.exists() else False:
        return "inside iCloud Drive"
    home = Path.home().resolve()
    for name in SYNCED_HOME_DIRS:
        if _under(resolved, home / name) and icloud_desktop_documents_enabled():
            return f"under ~/{name} with iCloud Desktop & Documents sync on"
    return None


def resolve_export_target(
    settings: Settings, out: Path | None, default_name: str, *, force: bool = False
) -> Path:
    """The path to write an export to; refuses cloud-synced targets unless ``force``."""
    if out is None:
        out_dir = Path(settings.data_dir) / EXPORTS_SUBDIR
        out_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(out_dir, 0o700)
        return out_dir / default_name
    target = Path(out).expanduser()
    reason = cloud_synced(target)
    if reason and not force:
        raise PraktikaError(
            f"refusing to write {target}: {reason}. Minutes and transcripts must not be "
            "cloud-synced (C-01); choose a location your organisation manages or pass --force."
        )
    home = Path.home().resolve()
    if any(_under(target.resolve(), home / n) for n in SYNCED_HOME_DIRS):
        log.warning("export.home_folder", path=str(target))
    return target
