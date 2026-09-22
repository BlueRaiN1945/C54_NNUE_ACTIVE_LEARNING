"""OS-neutral path helpers for the C54 Medium-PC audit control plane.

These conventions mirror -- but are implemented independently of -- the
'relative-neutral-path' format checker already frozen in Phase 0's
medium_pc_audit.schemas.registry. That module is not modified here; both
must agree in behavior on the rules they share. See
tests/test_phase0_neutral_path_equivalence.py for the regression guard.

Control-character rejection (below) is a Phase-1-only hardening addition,
not present in the frozen Phase 0 checker.
"""

from __future__ import annotations

import re
from pathlib import Path

_WINDOWS_ABS_RE = re.compile(r"^[A-Za-z]:[\\/]")


def require_neutral_path(value: str) -> str:
    """Validate value as a relative, OS-neutral, forward-slash path.

    Rejects: non-strings, empty strings, control characters (CR, LF, NUL,
    or any other C0/DEL byte -- so SHA256SUMS.txt, one path per line, can
    never become ambiguous), backslashes, a leading '/' (POSIX absolute), a
    Windows drive-letter prefix, and any '' or '..' segment.
    """
    if not isinstance(value, str) or not value:
        raise ValueError(f"neutral path must be a non-empty string, got {value!r}")
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value):
        raise ValueError(f"neutral path must not contain control characters, got {value!r}")
    if "\\" in value:
        raise ValueError(f"neutral path must not contain backslashes, got {value!r}")
    if value.startswith("/"):
        raise ValueError(f"neutral path must not be absolute, got {value!r}")
    if _WINDOWS_ABS_RE.match(value):
        raise ValueError(f"neutral path must not look like a Windows absolute path, got {value!r}")
    segments = value.split("/")
    if any(segment in ("", "..") for segment in segments):
        raise ValueError(f"neutral path must not contain empty or '..' segments, got {value!r}")
    return value


def to_neutral_path(file_path: Path, base: Path) -> str:
    """Compute file_path's path relative to base as a validated neutral path."""
    rel = Path(file_path).resolve().relative_to(Path(base).resolve())
    neutral = "/".join(rel.parts)
    return require_neutral_path(neutral)


def from_neutral_path(neutral: str, base: Path) -> Path:
    """Resolve a validated neutral path back to a real OS path under base."""
    require_neutral_path(neutral)
    return Path(base).joinpath(*neutral.split("/"))
