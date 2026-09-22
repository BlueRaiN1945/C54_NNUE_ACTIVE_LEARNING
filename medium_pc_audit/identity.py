"""Deterministic identity and canonicalization helpers for the C54 Medium-PC audit
control plane.

All functions here are pure: no filesystem access, no network access, no wall-clock
reads, and no dependence on OS-specific path formatting. Determinism is the whole
point -- the same logical inputs must always produce the same identifiers and the
same canonical bytes, on any operating system, forever.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class InvalidIdentityComponent(ValueError):
    """Raised when an identity component is missing, empty, or otherwise malformed.

    Identity components must fail loudly rather than silently produce a malformed
    filesystem-oriented identifier (e.g. an empty audit_id must never collapse into
    a run_id like "__run001").
    """


def _require_strict_int(value: Any, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidIdentityComponent(
            f"{name} must be a plain int (bool is not accepted), "
            f"got {type(value).__name__}: {value!r}"
        )
    if value < minimum:
        raise InvalidIdentityComponent(f"{name} must be >= {minimum}, got {value}")
    return value


def _require_identifier(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise InvalidIdentityComponent(f"{name} must be a non-empty string, got {value!r}")
    if not _IDENTIFIER_RE.fullmatch(value):
        raise InvalidIdentityComponent(
            f"{name} must match {_IDENTIFIER_RE.pattern!r} "
            f"(safe filesystem/path identifier), got {value!r}"
        )
    return value


def compute_audit_id(champion_version: int) -> str:
    """Build the audit identity for a champion version.

    Example: champion_version=11 -> 'AUDIT_V0011'.
    """
    _require_strict_int(champion_version, name="champion_version", minimum=0)
    return f"AUDIT_V{champion_version:04d}"


def compute_matchup_key(audit_id: str, opponent_id: str) -> str:
    """Compose the human-readable matchup key.

    Example: ('AUDIT_V0011', 'TEACHER_V0') -> 'AUDIT_V0011__TEACHER_V0'.
    """
    _require_identifier(audit_id, name="audit_id")
    _require_identifier(opponent_id, name="opponent_id")
    return f"{audit_id}__{opponent_id}"


def compute_run_id(matchup_key: str, attempt: int) -> str:
    """Compose the run identity for one execution attempt of a matchup."""
    _require_identifier(matchup_key, name="matchup_key")
    _require_strict_int(attempt, name="attempt", minimum=1)
    return f"{matchup_key}__run{attempt:03d}"


def canonicalize(obj: Mapping[str, Any], exclude_fields: frozenset = frozenset()) -> bytes:
    """Serialize a top-level mapping to deterministic canonical JSON bytes.

    Locked rules (do not change without updating every consumer of identity_sha256):
      - field exclusion is TOP-LEVEL ONLY; a nested field with the same name as an
        excluded top-level field is never touched.
      - sorted keys, compact fixed separators ("," and ":"), ensure_ascii=False,
        allow_nan=False.
      - UTF-8 encoded, exactly one trailing '\n'.
    """
    if not isinstance(obj, Mapping):
        raise TypeError(f"canonicalize() expects a top-level mapping, got {type(obj).__name__}")

    filtered = {key: value for key, value in obj.items() if key not in exclude_fields}

    text = json.dumps(
        filtered,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return (text + "\n").encode("utf-8")


def compute_identity_sha256(obj: Mapping[str, Any], exclude_fields: frozenset = frozenset()) -> str:
    """SHA256 hex digest of the canonical form of obj, with top-level exclude_fields removed first."""
    return hashlib.sha256(canonicalize(obj, exclude_fields)).hexdigest()
