"""Optional macOS sandbox-exec wrapper around the worker subprocess."""

from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def profile_path() -> Path:
    return Path(__file__).resolve().parents[3] / "sandbox" / "worker.sb"


def sandbox_enabled(flag: bool | None, cfg_default: bool) -> bool:
    env = os.environ.get("SLASHCOMPUTE_SANDBOX")
    if env is not None:
        return env.lower() in ("1", "true", "yes", "on")
    if flag is not None:
        return flag
    return bool(cfg_default) and sys.platform == "darwin"


def wrap_command(cmd: list[str], job_dir: Path, agent_dir: Path) -> list[str]:
    """Prefix ``cmd`` with sandbox-exec when the tool and profile exist."""
    exe = shutil.which("sandbox-exec")
    profile = profile_path()
    if exe is None or not profile.is_file():
        log.warning("sandbox requested but sandbox-exec/profile unavailable; running unsandboxed")
        return cmd
    tmp = Path(os.environ.get("TMPDIR") or "/tmp")
    return [
        exe, "-f", str(profile),
        "-D", f"JOBDIR={job_dir}",
        "-D", f"AGENTDIR={agent_dir}",
        "-D", f"TMPDIR={tmp}",
        "--", *cmd,
    ]
